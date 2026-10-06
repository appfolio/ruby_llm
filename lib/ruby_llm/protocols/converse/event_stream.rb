# frozen_string_literal: true

require 'base64'
require 'faraday'
require 'json'

module RubyLLM
  module Protocols
    class Converse
      # AWS Event Stream plumbing shared by every bedrock-runtime streaming endpoint: ConverseStream
      # (Converse::Streaming) and InvokeModelWithResponseStream (InvokeAnthropic::Streaming). Both
      # frame each event as a binary eventstream message whose JSON payload may wrap the real
      # event as base64 `bytes`, and both report mid-stream failures as `:exception-type`
      # messages. What each endpoint puts INSIDE the payload differs, so event interpretation
      # stays with the including protocol.
      module EventStream
        ErrorResponse = Struct.new(:body, :status)

        # Bedrock stream exception type => HTTP status, so ErrorMiddleware raises the same error
        # class a synchronous call with that failure would.
        EXCEPTION_STATUSES = {
          'throttlingException' => 429,
          'validationException' => 400,
          'accessDeniedException' => 401,
          'unrecognizedClientException' => 401,
          'serviceUnavailableException' => 503
        }.freeze

        private

        # POSTs a SigV4-signed eventstream request and hands every raw body chunk of a 200
        # response to the block. Non-200 chunks go through handle_failed_stream.
        def post_event_stream(url, payload, additional_headers = {}, &on_chunk)
          body = JSON.generate(payload)

          @connection.post(url, payload) do |req|
            req.headers.merge!(@provider.sign_headers('POST', url, body))
            req.headers.merge!(additional_headers) unless additional_headers.empty?
            req.headers['Accept'] = 'application/vnd.amazon.eventstream'

            if Faraday::VERSION.start_with?('1')
              req.options[:on_data] = proc { |chunk, _size| on_chunk.call(chunk) }
            else
              req.options.on_data = proc do |chunk, _bytes, env|
                if env&.status == 200
                  on_chunk.call(chunk)
                else
                  handle_failed_stream(chunk, env)
                end
              end
            end
          end
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

        # Every complete eventstream message in raw_chunk (the decoder buffers partial ones
        # across calls).
        def decode_event_messages(decoder, raw_chunk)
          messages = []
          message, eof = decoder.decode_chunk(raw_chunk)

          while message
            messages << message
            break if eof

            message, eof = decoder.decode_chunk
          end

          messages
        end

        def event_type_header(message)
          headers = message.headers
          header = headers[':event-type'] || headers[':exception-type']
          value = header.respond_to?(:value) ? header.value : header
          value.is_a?(String) && !value.empty? ? value : nil
        rescue StandardError
          nil
        end

        def exception_type_header(message)
          header = message.headers[':exception-type']
          value = header.respond_to?(:value) ? header.value : header
          value.is_a?(String) && !value.empty? ? value : nil
        rescue StandardError
          nil
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

        def raise_event_stream_exception(exception_type, message)
          status = EXCEPTION_STATUSES.fetch(exception_type, 500)
          response = ErrorResponse.new({ 'message' => message || exception_type }, status)
          ErrorMiddleware.parse_error(provider: self, response: response)
        end
      end
    end
  end
end
