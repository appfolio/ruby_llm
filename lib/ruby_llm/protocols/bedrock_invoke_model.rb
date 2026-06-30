# frozen_string_literal: true

module RubyLLM
  module Protocols
    # AWS Bedrock InvokeModel API – sends the raw Anthropic Messages wire format.
    # Standalone protocol (does not inherit Converse) so that Converse-era module
    # patches on the consuming app cannot fire on this path. SigV4 signing is
    # provided by the provider; the AWS event-stream decode is self-contained below.
    class BedrockInvokeModel < Protocol
      include BedrockInvokeModel::Chat
      include BedrockInvokeModel::Streaming

      private

      # Override the base Protocol#sync_response to add SigV4 signing, identical
      # to Converse#sync_response.
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
