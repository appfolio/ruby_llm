# frozen_string_literal: true

module RubyLLM
  module Protocols
    class Responses
      # Streaming methods of the OpenAI Responses API. Events are semantic:
      # each SSE data frame carries a `type` describing what changed.
      module Streaming
        module_function

        # GPT-6 only: streamed text deltas (commentary included) accumulate into content as they
        # arrive. Once the stream ends, a reply that needs its full item list (see
        # Chat#output_items?) gets it as OutputItems, the same content a sync reply would hold.
        def stream_response(payload, additional_headers = {}, &)
          return super unless gpt6_model?(model&.id)

          @streamed_output = nil
          @streamed_items = {}
          message = super
          output = streamed_output
          message.content = OutputItems.new(output) if output_items?(output)
          message
        end

        # response.completed carries the full output list; the output_item.done events are the
        # fallback if it ever arrives without one.
        def streamed_output
          return @streamed_output if @streamed_output.is_a?(Array) && !@streamed_output.empty?

          (@streamed_items || {}).sort_by(&:first).map(&:last)
        end

        def build_chunk(data)
          case data['type']
          when 'response.output_text.delta'
            chunk content: data['delta']
          when 'response.reasoning_summary_text.delta'
            chunk thinking: Thinking.build(text: data['delta'])
          when 'response.output_text.annotation.added'
            chunk citations: parse_annotations([data['annotation']], nil)
          when 'response.output_item.added'
            build_item_added_chunk(data)
          when 'response.function_call_arguments.delta'
            chunk tool_calls: { data['output_index'] => ToolCall.new(id: nil, name: nil, arguments: data['delta']) }
          when 'response.output_item.done'
            build_item_done_chunk(data)
          when 'response.completed'
            build_completed_chunk(data)
          else
            chunk
          end
        end

        def build_item_added_chunk(data)
          item = data['item']
          return chunk unless item['type'] == 'function_call'

          chunk tool_calls: {
            data['output_index'] => ToolCall.new(id: item['call_id'], name: item['name'], arguments: +'')
          }
        end

        def build_item_done_chunk(data)
          item = data['item']
          gpt6 = gpt6_model?(model&.id)
          @streamed_items[data['output_index']] = item if gpt6 && @streamed_items && !data['output_index'].nil?
          return chunk unless item['type'] == 'reasoning' && item['encrypted_content']
          return chunk thinking: Thinking.build(text: nil, signature: item['encrypted_content']) unless gpt6

          chunk thinking: Thinking.build(text: nil, signature: item['encrypted_content'], blocks: [item])
        end

        def build_completed_chunk(data)
          response = data['response'] || {}
          gpt6 = gpt6_model?(model&.id || response['model'])
          @streamed_output = response['output'] if gpt6

          chunk model_id: response['model'],
                finish_reason: response.dig('incomplete_details', 'reason'),
                **parse_usage(response['usage'] || {}, gpt6:)
        end

        def chunk(content: nil, **attributes)
          Chunk.new(role: :assistant, content: content, **attributes)
        end
      end
    end
  end
end
