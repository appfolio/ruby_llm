# frozen_string_literal: true

module RubyLLM
  module Protocols
    # Bedrock InvokeModel protocol — sends the raw Anthropic Messages format directly
    # to the bedrock-runtime InvokeModel endpoint. This unlocks anthropic_beta features
    # (e.g. context_management / server-side prompt caching) that are not available via
    # the Converse API. Purely additive: Converse remains the default path.
    class BedrockInvokeModel < Protocol
      include BedrockInvokeModel::Chat
      include BedrockInvokeModel::Streaming

      private

      def sync_response(payload, additional_headers = {})
        body = JSON.generate(payload)
        response = @connection.post(completion_url, payload) do |req|
          req.headers.merge!(@provider.sign_headers('POST', completion_url, body))
          req.headers.merge!(additional_headers) unless additional_headers.empty?
        end
        parse_completion_response(response)
      end
    end
  end
end
