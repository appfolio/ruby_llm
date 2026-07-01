# frozen_string_literal: true

module RubyLLM
  # Represents provider thinking output.
  class Thinking
    attr_reader :text, :signature, :blocks

    # blocks: the exact, provider-native reasoning content blocks for this turn, in
    # original order. Anthropic (and Bedrock Converse, which proxies to it) requires every
    # thinking/redacted_thinking block from the most recent assistant turn to be replayed
    # byte-for-byte, including blocks with empty thinking text — reconstructing a single
    # block from merged text+signature loses any additional blocks and gets rejected with
    # "Invalid data in redacted_thinking block" on the next request. When blocks is present,
    # protocols must replay it verbatim instead of rebuilding from text/signature.
    def initialize(text: nil, signature: nil, blocks: nil)
      @text = text
      @signature = signature
      @blocks = blocks
    end

    def self.build(text: nil, signature: nil, blocks: nil)
      text = presence(text)
      signature = presence(signature)
      blocks = presence(blocks)

      return nil if text.nil? && signature.nil? && blocks.nil?

      new(text: text, signature: signature, blocks: blocks)
    end

    def self.presence(value)
      value.nil? || value.empty? ? nil : value
    end
    private_class_method :presence

    def pretty_print(printer)
      printer.object_group(self) do
        printer.breakable
        printer.text 'text='
        printer.pp text
        printer.comma_breakable
        printer.text 'signature='
        printer.pp(signature ? '[REDACTED]' : nil)
      end
    end
  end

  class Thinking
    # Normalized config for thinking across providers.
    class Config
      attr_reader :effort, :budget

      def initialize(effort: nil, budget: nil)
        @effort = effort.is_a?(Symbol) ? effort.to_s : effort
        @budget = budget
      end

      def enabled?
        !effort.nil? || !budget.nil?
      end
    end
  end
end
