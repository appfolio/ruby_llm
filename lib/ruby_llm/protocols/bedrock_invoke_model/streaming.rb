# frozen_string_literal: true

require 'base64'
require 'faraday'
require 'json'

module RubyLLM
  module Protocols
    class BedrockInvokeModel
      # Streaming implementation for Bedrock InvokeModel with response stream
      # (AWS Event Stream). The event-stream byte decode below is duplicated from
      # Converse::Streaming intentionally — keeping the two paths byte-for-byte
      # independent ensures Converse patches never fire on the InvokeModel path.
      module Streaming
        ErrorResponse = Struct.new(:body, :status)

        private

        def stream_url
          "/model/#{escape_model_id(@model.id)}/invoke-with-response-stream"
        end

        def stream_response(payload, additional_headers = {}, &block)
          accumulator = StreamAccumulator.new
          decoder = event_stream_decoder
          thinking_state = {}
          body = JSON.generate(payload)

          response = @connection.post(stream_url, payload) do |req|
            req.headers.merge!(@provider.sign_headers('POST', stream_url, body))
            req.headers.merge!(additional_headers) unless additional_headers.empty?
            req.headers['Accept'] = 'application/vnd.amazon.eventstream'

            if Faraday::VERSION.start_with?('1')
              req.options[:on_data] = proc do |chunk, _size|
                parse_stream_chunk(decoder, chunk, accumulator, thinking_state, &block)
              end
            else
              req.options.on_data = proc do |chunk, _bytes, env|
                if env&.status == 200
                  parse_stream_chunk(decoder, chunk, accumulator, thinking_state, &block)
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

        def parse_stream_chunk(decoder, raw_chunk, accumulator, thinking_state)
          handle_non_eventstream_error_chunk(raw_chunk)

          decode_events(decoder, raw_chunk).each do |event|
            chunk = build_chunk(event, thinking_state)
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
                "Bedrock InvokeModel stream event keys: #{event.keys}"
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
          RubyLLM.logger.debug { "Failed to decode Bedrock InvokeModel stream event payload: #{e.message}" }
          nil
        end

        def build_chunk(event, thinking_state = {})
          raise_stream_error(event) if stream_error_event?(event)

          type = event['type']

          case type
          when 'message_start'
            build_message_start_chunk(event)
          when 'content_block_start'
            build_content_block_start_chunk(event, thinking_state)
          when 'content_block_delta'
            build_content_block_delta_chunk(event, thinking_state)
          when 'content_block_stop'
            build_content_block_stop_chunk(event, thinking_state)
          when 'message_delta'
            build_message_delta_chunk(event)
          else
            Chunk.new(role: :assistant, content: nil, model_id: @model&.id)
          end
        end

        def build_message_start_chunk(event)
          message = event['message'] || {}
          usage = message['usage'] || {}
          input_tok = usage['input_tokens']

          Chunk.new(
            role: :assistant,
            content: nil,
            model_id: message['model'] || @model&.id,
            input_tokens: input_tok ? [input_tok.to_i, 0].max : nil
          )
        end

        def build_content_block_start_chunk(event, thinking_state)
          content_block = event['content_block'] || {}
          index = event['index']
          tool_calls = nil
          thinking = nil

          case content_block['type']
          when 'tool_use'
            id = content_block['id']
            tool_calls = {
              id => ToolCall.new(id: id, name: content_block['name'], arguments: {})
            }
          when 'redacted_thinking'
            thinking = Thinking.build(blocks: [{ 'type' => 'redacted_thinking', 'data' => content_block['data'] }])
          when 'thinking'
            thinking_state[index] = { text: +'', signature: nil }
          end

          Chunk.new(
            role: :assistant,
            content: nil,
            model_id: @model&.id,
            thinking: thinking,
            tool_calls: tool_calls
          )
        end

        def build_content_block_delta_chunk(event, thinking_state)
          delta = event['delta'] || {}
          delta_type = delta['type']
          index = event['index']

          content = nil
          thinking_text = nil
          thinking_sig = nil
          tool_calls = nil

          case delta_type
          when 'text_delta'
            content = delta['text']
          when 'input_json_delta'
            partial = delta['partial_json']
            tool_calls = { nil => ToolCall.new(id: nil, name: nil, arguments: partial) } if partial
          when 'thinking_delta'
            thinking_text = delta['thinking']
            thinking_state[index][:text] << thinking_text.to_s if thinking_state[index]
          when 'signature_delta'
            thinking_sig = delta['signature']
            thinking_state[index][:signature] = thinking_sig if thinking_state[index]
          end

          Chunk.new(
            role: :assistant,
            model_id: @model&.id,
            content: content,
            thinking: Thinking.build(text: thinking_text, signature: thinking_sig),
            tool_calls: tool_calls
          )
        end

        # A thinking block only finalizes here, on its content_block_stop. If the turn
        # is truncated (e.g. stop_reason 'max_tokens') before this event arrives for a
        # given index, that block is intentionally dropped rather than replayed
        # signature-less — Anthropic rejects replay of a thinking block without a valid
        # signature, so a half-formed block is worse than none on the next request.
        def build_content_block_stop_chunk(event, thinking_state)
          index = event['index']
          state = thinking_state.delete(index)
          return Chunk.new(role: :assistant, content: nil, model_id: @model&.id) unless state

          block = { 'type' => 'thinking', 'thinking' => state[:text], 'signature' => state[:signature] }

          Chunk.new(
            role: :assistant,
            content: nil,
            model_id: @model&.id,
            thinking: Thinking.build(blocks: [block])
          )
        end

        def build_message_delta_chunk(event)
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
      end
    end
  end
end
