# frozen_string_literal: true

module RubyLLM
  # Represents token usage for a response.
  class Tokens
    attr_reader :input, :output, :cached, :cache_creation, :thinking,
                :cache_creation_ephemeral_5m, :cache_creation_ephemeral_1h

    # rubocop:disable Metrics/ParameterLists
    def initialize(input: nil, output: nil, cached: nil, cache_creation: nil, thinking: nil,
                   cache_creation_ephemeral_5m: nil, cache_creation_ephemeral_1h: nil)
      @input = input
      @output = output
      @cached = cached
      @cache_creation = cache_creation
      @thinking = thinking
      @cache_creation_ephemeral_5m = cache_creation_ephemeral_5m
      @cache_creation_ephemeral_1h = cache_creation_ephemeral_1h
    end
    # rubocop:enable Metrics/ParameterLists

    # rubocop:disable Metrics/ParameterLists
    def self.build(input: nil, output: nil, cached: nil, cache_creation: nil, thinking: nil,
                   cache_creation_ephemeral_5m: nil, cache_creation_ephemeral_1h: nil)
      return nil if [input, output, cached, cache_creation, thinking,
                     cache_creation_ephemeral_5m, cache_creation_ephemeral_1h].all?(&:nil?)

      new(
        input: input,
        output: output,
        cached: cached,
        cache_creation: cache_creation,
        thinking: thinking,
        cache_creation_ephemeral_5m: cache_creation_ephemeral_5m,
        cache_creation_ephemeral_1h: cache_creation_ephemeral_1h
      )
    end
    # rubocop:enable Metrics/ParameterLists

    def to_h
      {
        input_tokens: input,
        output_tokens: output,
        cached_tokens: cached,
        cache_creation_tokens: cache_creation,
        thinking_tokens: thinking,
        cache_creation_ephemeral_5m_tokens: cache_creation_ephemeral_5m,
        cache_creation_ephemeral_1h_tokens: cache_creation_ephemeral_1h
      }.compact
    end

    def cache_read
      cached
    end

    def cache_write
      cache_creation
    end
  end
end
