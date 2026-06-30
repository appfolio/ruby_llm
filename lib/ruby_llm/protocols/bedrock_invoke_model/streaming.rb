# frozen_string_literal: true

require 'base64'
require 'faraday'
require 'json'

module RubyLLM
  module Protocols
    class BedrockInvokeModel
      # Streaming for Bedrock InvokeModel with Response Stream.
      # Self-contained: does NOT include or inherit from Converse::Streaming so that
      # consuming-app patches on that module cannot fire on this path.
      # The AWS event-stream byte decode (decode_events / decode_event_payload) is
      # duplicated here intentionally — see bedrock_invoke_model.rb for the rationale.
      module Streaming
        # re-verify on gem bump: Struct used to mirror Converse::Streaming::ErrorResponse
        ErrorResponse = Struct.new(:body, :status)

        private

        def stream_url
          "/model/#{escape_model_id(@model.id)}/invoke-with-response-stream"
        end

        def stream_response(payload, additional_headers = {}, &block)
          accumulator = StreamAccumulator.new
          decoder = event_stream_decoder
          body = JSON.generate(payload)

          response = @connection.post(stream_url, payload) do |req|
            req.headers.merge!(@provider.sign_headers('POST', stream_url, body))
            req.headers.merge!(additional_headers) unless additional_headers.empty?
            req.headers['Accept'] = 'application/vnd.amazon.eventstream'

            if Faraday::VERSION.start_with?('1')
              req.options[:on_data] = proc do |chunk, _size|
                parse_stream_chunk(decoder, chunk, accumulator, &block)
              end
            else
              req.options.on_data = proc do |chunk, _bytes, env|
                if env&.status == 200
                  parse_stream_chunk(decoder, chunk, accumulator, &block)
                else
                  handle_failed_stream(chunk, env)
                end
              end
            end
          end

          message = accumulator.to_message(response)
          RubyLLM.logger.debug { "Stream completed: #{message.content}" }
          message
        end

        def event_stream_decoder
          require 'aws-eventstream'
          Aws::EventStream::Decoder.new
        rescue LoadError
          raise Error,
                'The aws-eventstream gem is required for Bedrock streaming. ' \
                'Please add it to your Gemfile: gem "aws-eventstream"'
        end

        def handle_failed_stream(chunk, env)
          data = JSON.parse(chunk)
          error_response = env.merge(body: data)
          ErrorMiddleware.parse_error(provider: self, response: error_response)
        rescue JSON::ParserError
          RubyLLM.logger.debug { "Failed Bedrock stream error chunk: #{chunk}" }
        end

        def parse_stream_chunk(decoder, raw_chunk, accumulator)
          handle_non_eventstream_error_chunk(raw_chunk)

          decode_events(decoder, raw_chunk).each do |event|
            chunk = build_chunk(event)
            next unless chunk

            accumulator.add(chunk)
            yield chunk
          end
        end

        def handle_non_eventstream_error_chunk(raw_chunk)
          text = raw_chunk.to_s

          if text.start_with?('event: error')
            payload = text.lines.find { |line| line.start_with?('data:') }&.delete_prefix('data:')&.strip
            raise_streaming_chunk_error(payload) if payload
            return
          end

          return unless text.lstrip.start_with?('{') && text.include?('"error"')

          raise_streaming_chunk_error(text)
        end

        def raise_streaming_chunk_error(payload)
          parsed = JSON.parse(payload)
          message = parsed.dig('error', 'message') || parsed['message'] || 'Bedrock streaming error'
          response = ErrorResponse.new({ 'message' => message }, 500)
          ErrorMiddleware.parse_error(provider: self, response: response)
        rescue JSON::ParserError
          nil
        end

        # re-verify on gem bump: aws-eventstream Decoder#decode_chunk API
        def decode_events(decoder, raw_chunk)
          events = []
          message, eof = decoder.decode_chunk(raw_chunk)

          while message
            event = decode_event_payload(message.payload.read)
            if event && RubyLLM.config.log_stream_debug
              RubyLLM.logger.debug do
                "Bedrock stream event keys: #{event.keys}"
              end
            end
            events << event if event
            break if eof

            message, eof = decoder.decode_chunk
          end

          events
        end

        def decode_event_payload(payload)
          outer = JSON.parse(payload)

          if outer['bytes'].is_a?(String)
            JSON.parse(Base64.decode64(outer['bytes']))
          else
            outer
          end
        rescue JSON::ParserError => e
          RubyLLM.logger.debug { "Failed to decode Bedrock stream event payload: #{e.message}" }
          nil
        end

        def build_chunk(event)
          raise_stream_error(event) if stream_error_event?(event)

          event_type = event['type']

          case event_type
          when 'message_start'
            build_invoke_message_start_chunk(event)
          when 'content_block_start'
            build_invoke_content_block_start_chunk(event)
          when 'content_block_delta'
            build_invoke_content_block_delta_chunk(event)
          when 'content_block_stop', 'message_stop'
            nil
          when 'message_delta'
            build_invoke_message_delta_chunk(event)
          end
        end

        def stream_error_event?(event)
          event.keys.any? { |key| key.end_with?('Exception') } || event['type'] == 'error'
        end

        def raise_stream_error(event)
          if event['type'] == 'error'
            message = event.dig('error', 'message') || 'Bedrock streaming error'
            response = ErrorResponse.new({ 'message' => message }, 500)
            ErrorMiddleware.parse_error(provider: self, response: response)
            return
          end

          key = event.keys.find { |candidate| candidate.end_with?('Exception') }
          payload = event[key]
          message = payload['message'] || key
          status = case key
                   when 'throttlingException' then 429
                   when 'validationException' then 400
                   when 'accessDeniedException', 'unrecognizedClientException' then 401
                   when 'serviceUnavailableException' then 503
                   else 500
                   end

          response = ErrorResponse.new({ 'message' => message }, status)
          ErrorMiddleware.parse_error(provider: self, response: response)
        end

        def build_invoke_message_start_chunk(event)
          message = event['message'] || {}
          usage = message['usage'] || {}
          input = usage['input_tokens']
          Chunk.new(
            role: :assistant,
            content: nil,
            model_id: message['model'] || @model&.id,
            input_tokens: input ? [input.to_i, 0].max : nil
          )
        end

        def build_invoke_content_block_start_chunk(event)
          content_block = event['content_block'] || {}
          return nil unless content_block['type'] == 'tool_use'

          id = content_block['id']
          index = event['index']
          Chunk.new(
            role: :assistant,
            content: nil,
            model_id: @model&.id,
            tool_calls: {
              index => ToolCall.new(id: id, name: content_block['name'], arguments: {})
            }
          )
        end

        def build_invoke_content_block_delta_chunk(event)
          delta = event['delta'] || {}
          delta_type = delta['type']

          case delta_type
          when 'text_delta'
            Chunk.new(role: :assistant, content: delta['text'], model_id: @model&.id)
          when 'input_json_delta'
            index = event['index']
            Chunk.new(
              role: :assistant,
              content: nil,
              model_id: @model&.id,
              tool_calls: { index => ToolCall.new(id: nil, name: nil, arguments: delta['partial_json'].to_s) }
            )
          when 'thinking_delta'
            Chunk.new(
              role: :assistant,
              content: nil,
              model_id: @model&.id,
              thinking: Thinking.build(text: delta['thinking'])
            )
          when 'signature_delta'
            Chunk.new(
              role: :assistant,
              content: nil,
              model_id: @model&.id,
              thinking: Thinking.build(signature: delta['signature'])
            )
          end
        end

        def build_invoke_message_delta_chunk(event)
          delta = event['delta'] || {}
          usage = event['usage'] || {}
          Chunk.new(
            role: :assistant,
            content: nil,
            model_id: @model&.id,
            output_tokens: usage['output_tokens'],
            finish_reason: delta['stop_reason']
          )
        end
      end
    end
  end
end
