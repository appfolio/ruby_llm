# frozen_string_literal: true

require 'json'

module RubyLLM
  module Protocols
    class InvokeAnthropic
      # Streaming for Bedrock InvokeModelWithResponseStream. The wire format is an AWS
      # eventstream (decoded by Converse::EventStream) whose `chunk` events each carry one
      # Anthropic Messages streaming event as base64 `bytes`; the decoded event is then
      # interpreted exactly as the first-party Anthropic SSE stream would be, plus raw thinking
      # block capture for verbatim replay, the full block list for replies that need it (see
      # Chat#reply_content) and provider_data.
      #
      # Event formats recorded from Bedrock (spec/ruby_llm/protocols/invoke_anthropic_server_features_spec.rb):
      # - compaction: content_block_start with `content: null`, then one compaction_delta
      #   carrying the whole summary in `content`, then content_block_stop.
      # - server_tool_use: content_block_start with `input: {}`, then input_json_delta events,
      #   like tool_use.
      # - tool_search_tool_result: complete in content_block_start, no deltas.
      # - message_start carries input_transformations and the pre-compaction input_tokens;
      #   message_delta carries context_management, usage.iterations and the final input_tokens.
      module Streaming
        include Converse::EventStream

        private

        def stream_url
          "/model/#{escape_model_id(@model.id)}/invoke-with-response-stream"
        end

        def stream_response(payload, additional_headers = {}, &block)
          accumulator = StreamAccumulator.new
          decoder = event_stream_decoder
          thinking_state = {}
          block_state = {}

          response = post_event_stream(stream_url, payload, additional_headers) do |raw_chunk|
            parse_stream_chunk(decoder, raw_chunk, accumulator, thinking_state, block_state, &block)
          end

          message = accumulator.to_message(response)
          apply_streamed_blocks(message, block_state)
          RubyLLM.logger.debug { "Stream completed: #{message.content}" }
          message
        end

        def parse_stream_chunk(decoder, raw_chunk, accumulator, thinking_state, block_state)
          handle_non_eventstream_error_chunk(raw_chunk)

          decode_event_messages(decoder, raw_chunk).each do |message|
            data = decode_invoke_event(message)
            next unless data

            track_content_block(data, block_state)
            chunk = build_chunk(data, thinking_state)
            accumulator.add(chunk)
            yield chunk
          end
        end

        # Returns the Anthropic event carried by one eventstream message, or raises for a
        # Bedrock exception message or an Anthropic `error` event.
        def decode_invoke_event(message)
          payload = message.payload.read
          exception_type = exception_type_header(message)
          if exception_type
            parsed = decode_event_payload(payload) || {}
            raise_event_stream_exception(exception_type, parsed['message'] || parsed['Message'])
          end

          data = decode_event_payload(payload)
          return nil unless data.is_a?(Hash)

          RubyLLM.logger.debug { "Bedrock invoke stream event: #{data['type']}" } if RubyLLM.config.log_stream_debug
          raise_anthropic_stream_error(data) if data['type'] == 'error'
          data
        end

        def raise_anthropic_stream_error(data)
          status = data.dig('error', 'type') == 'overloaded_error' ? 529 : 500
          message = data.dig('error', 'message') || 'Bedrock streaming error'
          response = Converse::EventStream::ErrorResponse.new({ 'message' => message }, status)
          ErrorMiddleware.parse_error(provider: self, response: response)
        end

        def build_chunk(data, thinking_state = {})
          delta_type = data.dig('delta', 'type')
          track_thinking_block(data, delta_type, thinking_state)

          Chunk.new(
            role: :assistant,
            model_id: extract_model_id(data),
            content: extract_content_delta(data, delta_type),
            citations: extract_citations_delta(data, delta_type),
            thinking: Thinking.build(
              text: extract_thinking_delta(data, delta_type),
              signature: extract_signature_delta(data, delta_type),
              blocks: extract_finalized_thinking_blocks(data, thinking_state)
            ),
            input_tokens: extract_input_tokens(data),
            output_tokens: extract_output_tokens(data),
            cached_tokens: extract_cached_tokens(data),
            cache_creation_tokens: extract_cache_creation_tokens(data),
            tool_calls: extract_tool_calls(data),
            finish_reason: data.dig('delta', 'stop_reason'),
            provider_data: provider_data_for(data['message'] || data)
          )
        end

        # message_delta's usage.input_tokens is the final count (after compaction, and after any
        # server-tool iterations); message_start's is taken before either. Matches the sync
        # response's top-level usage.input_tokens.
        def extract_input_tokens(data)
          data.dig('usage', 'input_tokens') || super
        end

        # Rebuilds every content block from its start event and deltas, keyed by index, in the
        # format the sync response returns.
        def track_content_block(data, block_state)
          case data['type']
          when 'content_block_start'
            block_state[data['index']] = { block: RubyLLM::Utils.deep_dup(data['content_block'] || {}), json: +'' }
          when 'content_block_delta'
            state = block_state[data['index']]
            apply_block_delta(state, data['delta'] || {}) if state
          when 'content_block_stop'
            state = block_state[data['index']]
            state[:block]['input'] = JSON.parse(state[:json]) if state && !state[:json].empty?
          end
        end

        def apply_block_delta(state, delta)
          block = state[:block]
          case delta['type']
          when 'text_delta' then block['text'] = block['text'].to_s + delta['text'].to_s
          when 'thinking_delta' then block['thinking'] = block['thinking'].to_s + delta['thinking'].to_s
          when 'signature_delta' then block['signature'] = delta['signature']
          when 'input_json_delta' then state[:json] << delta['partial_json'].to_s
          when 'citations_delta' then (block['citations'] ||= []) << delta['citation']
          when 'compaction_delta' then block['content'] = block['content'].to_s + delta['content'].to_s
          end
        end

        def apply_streamed_blocks(message, block_state)
          blocks = block_state.keys.sort.map { |index| block_state[index][:block] }
          message.content = ContentBlocks.new(blocks) unless modeled_blocks?(blocks)
        end

        # Thinking blocks arrive as content_block_start (type thinking / redacted_thinking,
        # the latter carrying its opaque `data` whole), then thinking_delta / signature_delta
        # events for the same index, then content_block_stop. State is keyed by index and
        # finalized into the exact block shape the non-streaming response returns.
        def track_thinking_block(data, delta_type, thinking_state)
          index = data['index']
          return if index.nil?

          if data['type'] == 'content_block_start'
            start_thinking_state(data['content_block'] || {}, index, thinking_state)
            return
          end

          state = thinking_state[index]
          return unless state && data['type'] == 'content_block_delta'

          case delta_type
          when 'thinking_delta' then state[:text] << data.dig('delta', 'thinking').to_s
          when 'signature_delta'
            state[:signature] = stream_presence(data.dig('delta', 'signature')) || state[:signature]
          end
        end

        def start_thinking_state(block, index, thinking_state)
          case block['type']
          when 'thinking'
            thinking_state[index] = {
              text: +block['thinking'].to_s, signature: stream_presence(block['signature']), redacted: nil
            }
          when 'redacted_thinking'
            thinking_state[index] = { text: +'', signature: nil, redacted: block['data'] }
          end
        end

        # A block finalizes on its own content_block_stop. As in Converse::Streaming, message_stop
        # also finalizes any still-open block that is structurally complete (signed or
        # redacted); a half-formed block is dropped because replaying it would be rejected.
        def extract_finalized_thinking_blocks(data, thinking_state)
          case data['type']
          when 'content_block_stop'
            state = thinking_state.delete(data['index'])
            block = state && finalize_thinking_block(state)
            block ? [block] : nil
          when 'message_stop'
            finalize_remaining_thinking_blocks(thinking_state)
          end
        end

        def finalize_remaining_thinking_blocks(thinking_state)
          blocks = thinking_state.keys.sort.filter_map do |index|
            state = thinking_state.delete(index)
            next unless state[:redacted] || state[:signature]

            finalize_thinking_block(state)
          end
          blocks.empty? ? nil : blocks
        end

        def finalize_thinking_block(state)
          return { 'type' => 'redacted_thinking', 'data' => state[:redacted] } if state[:redacted]
          return nil if state[:text].empty? && state[:signature].nil?

          { 'type' => 'thinking', 'thinking' => state[:text], 'signature' => state[:signature] }.compact
        end

        def stream_presence(value)
          value.is_a?(String) && !value.empty? ? value : nil
        end
      end
    end
  end
end
