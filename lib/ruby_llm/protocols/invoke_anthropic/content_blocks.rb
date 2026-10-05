# frozen_string_literal: true

module RubyLLM
  module Protocols
    class InvokeAnthropic
      # The complete Anthropic content-block list of a reply that holds a block the fork does
      # not otherwise model (compaction, server_tool_use, tool_search_tool_result, ...). As a
      # Content::Raw, an assistant message carrying it is sent back block-for-block on the next
      # InvokeModel request; #text (and #to_s) return the reply's text blocks joined, the same
      # string a plain reply would have as its content.
      class ContentBlocks < RubyLLM::Content::Raw
        def text
          value.filter_map { |block| block['text'] if block['type'] == 'text' }.join
        end

        alias to_s text
      end
    end
  end
end
