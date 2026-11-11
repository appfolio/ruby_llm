# frozen_string_literal: true

module RubyLLM
  module Providers
    class Bedrock
      module Mantle
        # The OpenAI Responses API under the openai/ prefix on the
        # bedrock-mantle endpoint, the only path that serves the GPT models
        # matching OPENAI_RESPONSES_MODEL_PATTERN.
        class OpenAIResponses < Responses
          def completion_url
            'openai/v1/responses'
          end
        end
      end
    end
  end
end
