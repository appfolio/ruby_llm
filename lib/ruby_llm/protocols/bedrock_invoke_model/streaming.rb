# frozen_string_literal: true

module RubyLLM
  module Protocols
    class BedrockInvokeModel
      # Streaming for Bedrock InvokeModel with Response Stream.
      # The AWS event-stream framing (decode_events / decode_event_payload / parse_stream_chunk)
      # is inherited from Converse::Streaming unchanged; only stream_url and build_chunk
      # are overridden to parse Anthropic SSE event types.
      module Streaming
        private

        def stream_url
          "/model/#{escape_model_id(@model.id)}/invoke-with-response-stream"
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
