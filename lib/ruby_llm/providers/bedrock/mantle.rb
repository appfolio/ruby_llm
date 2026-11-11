# frozen_string_literal: true

module RubyLLM
  module Providers
    class Bedrock
      # Shared plumbing for protocols served by the bedrock-mantle endpoint.
      # Mantle speaks the vendors' own wire formats rather than Converse, so
      # each protocol subclasses its family and mixes this in to borrow the
      # provider's mantle connection and SigV4-sign every request for the
      # bedrock-mantle service.
      module Mantle
        SIGNING_SERVICE = 'bedrock-mantle'

        # Mantle serves Claude on the Anthropic Messages API and everything
        # else on one of the two OpenAI surfaces. Only these models answer on
        # v1/responses; the rest of the non-Claude catalog answers on
        # v1/chat/completions and rejects v1/responses outright.
        RESPONSES_MODELS = %w[
          google.gemma-4-26b-a4b
          google.gemma-4-31b
          google.gemma-4-e2b
          openai.gpt-oss-20b
          openai.gpt-oss-120b
        ].freeze

        # These GPT models answer only on openai/v1/responses: openai.gpt-5.6-sol,
        # -terra and -luna, openai.gpt-6-sol, -luna and -astra, and
        # openai.gpt-6.1-sol and -terra. openai.gpt-5.5, openai.gpt-5.4 and
        # openai.gpt-oss-* are deliberately outside it; mantle serves them on
        # other paths or not at all.
        OPENAI_RESPONSES_MODEL_PATTERN = /\Aopenai\.gpt-(?:5\.6|6(?:\.\d+)?)-/

        def initialize(provider, model = nil)
          super
          @connection = provider.mantle_connection
        end

        private

        def sync_response(payload, additional_headers = {})
          parse_completion_response signed_post(completion_url, payload, additional_headers)
        end

        def signed_post(url, payload, additional_headers = {})
          body = JSON.generate(payload)

          @connection.post(url, payload, usage: @usage_tracker) do |req|
            req.headers.merge!(mantle_headers(url, body))
            req.headers.merge!(additional_headers) unless additional_headers.empty?
          end
        end

        def stream_response(payload, additional_headers = {}, &)
          body = JSON.generate(payload)
          super(payload, mantle_headers(stream_url, body).merge(additional_headers), &)
        end

        def mantle_headers(url, body)
          @provider.sign_headers('POST', url, body, base_url: @provider.mantle_api_base, service: SIGNING_SERVICE,
                                                    region: @provider.mantle_region)
        end
      end
    end
  end
end
