# frozen_string_literal: true

require 'json'

module RubyLLM
  module Protocols
    class Converse
      # Streaming implementation for Bedrock ConverseStream (AWS Event Stream).
      module Streaming
        include Bedrock::EventStream

        private

        def stream_url
          "/model/#{escape_model_id(@model.id)}/converse-stream"
        end

        def stream_response(payload, additional_headers = {}, &)
          @thinking_stream = ThinkingStream.new
          eventstream_response(payload, additional_headers, &)
        end

        def build_chunk(event)
          raise_stream_error(event) if stream_error_event?(event)

          metadata_usage, usage = event_usage(event)

          Chunk.new(
            role: :assistant,
            model: event['modelId'] || @model&.id,
            content: extract_content_delta(event),
            citations: extract_citations_delta(event),
            thinking: Thinking.build(
              text: extract_thinking_delta(event),
              signature: extract_thinking_signature(event)
            ),
            raw_reasoning: streamed_thinking_blocks(event),
            tool_calls: extract_tool_calls(event),
            server_tool_calls: extract_server_tool_call_events(event),
            input_tokens: extract_input_tokens(metadata_usage, usage),
            output_tokens: extract_output_tokens(metadata_usage, usage),
            cache_read_tokens: extract_cache_read_tokens(metadata_usage, usage),
            cache_write_tokens: extract_cache_write_tokens(metadata_usage, usage),
            thinking_tokens: extract_reasoning_tokens(metadata_usage, usage),
            finish_reason: extract_finish_reason(event)
          )
        end

        def streamed_thinking_blocks(event)
          @thinking_stream ||= ThinkingStream.new
          @thinking_stream.add(event)
          @thinking_stream.raw_reasoning if event.key?('messageStop') || event.key?('stopReason')
        end

        def extract_finish_reason(event)
          normalize_finish_reason(event.dig('messageStop', 'stopReason') || event['stopReason'])
        end

        def event_usage(event)
          [
            event.dig('metadata', 'usage') || {},
            event['usage'] || {}
          ]
        end

        def extract_input_tokens(metadata_usage, usage)
          bedrock_usage = metadata_usage['inputTokens'] ? metadata_usage : usage
          Chat.input_tokens(bedrock_usage) if bedrock_usage['inputTokens']
        end

        def extract_output_tokens(metadata_usage, usage)
          metadata_usage['outputTokens'] || usage['outputTokens']
        end

        def extract_cache_read_tokens(metadata_usage, usage)
          metadata_usage['cacheReadInputTokens'] || usage['cacheReadInputTokens']
        end

        def extract_cache_write_tokens(metadata_usage, usage)
          metadata_usage['cacheWriteInputTokens'] || usage['cacheWriteInputTokens']
        end

        def extract_reasoning_tokens(metadata_usage, usage)
          Chat.reasoning_tokens(metadata_usage) || Chat.reasoning_tokens(usage)
        end

        def extract_content_delta(event)
          normalized_delta(event)['text']
        end

        # The cited span itself streams as ordinary text deltas; the citation
        # delta carries the source. The final message resolves the span text
        # from the accumulated content when indices are known.
        def extract_citations_delta(event)
          citation = normalized_delta(event)['citation']
          return nil unless citation.is_a?(Hash)

          [Chat.parse_citation(citation)]
        end

        def extract_thinking_delta(event)
          reasoning_content = normalized_delta(event)['reasoningContent'] || {}

          reasoning_text = reasoning_content['reasoningText'] || {}
          return reasoning_text['text'] if reasoning_text.key?('text')
          return reasoning_content['text'] if reasoning_content.key?('text')
          return '' if [reasoning_text, reasoning_content].any? { |block| block.key?('signature') }

          extract_thinking_from_start(event)
        end

        def extract_thinking_from_start(event)
          start = event.dig('contentBlockStart', 'start', 'reasoningContent') || {}
          return unless start.key?('reasoningText')

          start.dig('reasoningText', 'text').to_s
        end

        def extract_thinking_signature(event)
          signature = extract_signature_from_delta(event)
          return signature if signature

          signature = extract_signature_from_start(event)
          return signature if signature

          nil
        end

        def extract_signature_from_delta(event)
          reasoning_content = normalized_delta(event)['reasoningContent'] || {}
          reasoning_text = reasoning_content['reasoningText'] || {}
          reasoning_text['signature'] || reasoning_content['signature'] || reasoning_content['redactedContent']
        end

        def extract_signature_from_start(event)
          start = event.dig('contentBlockStart', 'start', 'reasoningContent')
          return nil unless start

          reasoning_text = start['reasoningText'] || {}
          return reasoning_text['signature'] if reasoning_text['signature']
          return start['redactedContent'] if start['redactedContent']

          nil
        end

        def extract_tool_calls(event)
          return extract_tool_call_start(event) if tool_call_start_event?(event)
          return extract_tool_call_delta(event) if tool_call_delta_event?(event)

          nil
        end

        def tool_call_start_event?(event)
          event['contentBlockStart'] || event['start']
        end

        def tool_call_delta_event?(event)
          event['contentBlockDelta'] || event.dig('delta', 'toolUse')
        end

        def extract_tool_call_start(event)
          tool_use = event.dig('contentBlockStart', 'start', 'toolUse') || event.dig('start', 'toolUse')
          return nil unless tool_use

          if Chat.server_tool_use?(tool_use)
            remember_server_tool_block(event)
            return nil
          end

          tool_use_id = tool_use['toolUseId']
          {
            tool_use_id => ToolCall.new(
              id: tool_use_id,
              name: tool_use['name'],
              arguments: tool_use['input'] || {}
            )
          }
        end

        def extract_tool_call_delta(event)
          return nil if server_tool_block_event?(event)

          input = normalized_delta(event).dig('toolUse', 'input')
          return nil unless input

          { nil => ToolCall.new(id: nil, name: nil, arguments: input) }
        end

        # Server-executed tool steps stream as typed toolUse and toolResult
        # block starts. Their input deltas are not function-call arguments,
        # so the block index is remembered and its deltas skipped.
        def extract_server_tool_call_events(event)
          start = event.dig('contentBlockStart', 'start') || event['start']
          return [] unless start.is_a?(Hash)

          tool_use = start['toolUse']
          tool_result = start['toolResult']
          if tool_use && Chat.server_tool_use?(tool_use)
            [ServerToolCall.new(type: tool_use['type'], name: tool_use['name'], id: tool_use['toolUseId'],
                                input: tool_use['input'], raw: start)]
          elsif tool_result && Chat.server_tool_result?(tool_result)
            [ServerToolCall.new(type: tool_result['type'], id: tool_result['toolUseId'],
                                result: tool_result['content'], raw: start)]
          else
            []
          end
        end

        def remember_server_tool_block(event)
          index = event_block_index(event)
          return if index.nil?

          @server_tool_block_indices ||= {}
          @server_tool_block_indices[index] = true
        end

        def server_tool_block_event?(event)
          index = event_block_index(event)
          !index.nil? && @server_tool_block_indices&.key?(index)
        end

        def event_block_index(event)
          event['contentBlockIndex'] ||
            event.dig('contentBlockStart', 'contentBlockIndex') ||
            event.dig('contentBlockDelta', 'contentBlockIndex')
        end

        def normalized_delta(event)
          delta = event.dig('contentBlockDelta', 'delta') || event['delta'] || {}
          return delta if delta.is_a?(Hash)

          if delta.is_a?(String) && !delta.empty?
            JSON.parse(delta)
          else
            {}
          end
        rescue JSON::ParserError
          {}
        end
      end
    end
  end
end
