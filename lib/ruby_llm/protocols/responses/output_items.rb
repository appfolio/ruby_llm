# frozen_string_literal: true

module RubyLLM
  module Protocols
    class Responses
      # The complete, ordered output item list of a Responses reply, kept when the reply holds
      # items RubyLLM does not model (e.g. compaction) or interim commentary messages. The next
      # request replays it item-for-item. Also holds the response.compaction object returned by
      # the standalone /compact endpoint, whose items live under its 'output' field.
      class OutputItems < RubyLLM::Content::Raw
        # The reply's own answer: text of message items whose phase is final_answer or absent.
        def text
          message_text { |phase| phase != 'commentary' }
        end

        # Text of message items with phase: "commentary" (interim narration before tool calls).
        def commentary_text
          message_text { |phase| phase == 'commentary' }
        end

        alias to_s text

        def items
          value.is_a?(Hash) ? Array(value['output']) : Array(value)
        end

        private

        def message_text
          items.select { |item| item['type'] == 'message' && yield(item['phase']) }.flat_map do |item|
            Array(item['content']).filter_map { |part| part['text'] if part['type'] == 'output_text' }
          end.join
        end
      end
    end
  end
end
