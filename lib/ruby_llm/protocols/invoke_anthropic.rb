# frozen_string_literal: true

module RubyLLM
  module Protocols
    # AWS Bedrock InvokeModel / InvokeModelWithResponseStream for Anthropic Claude models,
    # speaking the Anthropic Messages request/response format natively instead of Bedrock's
    # Converse translation layer. Reaches Messages-API features Converse doesn't expose
    # (anthropic_beta headers-in-body, context_management, cache_control placement) without
    # leaving bedrock-runtime or SigV4.
    #
    # Differences from the first-party Anthropic protocol, all on the envelope:
    # - POST /model/{id}/invoke (and /invoke-with-response-stream), SigV4-signed by the provider.
    # - The body carries anthropic_version: "bedrock-2023-05-31" and never `model` or `stream`
    #   (the model is in the path; streaming is chosen by endpoint).
    # - The stream is an AWS eventstream whose `chunk` events wrap each Anthropic SSE event
    #   as base64 `bytes` (see Converse::EventStream), not an SSE body.
    #
    # Never the Bedrock default: selected per request with `protocol: :invoke_anthropic`
    # (RubyLLM::Chat#with_protocol) or config.bedrock_protocol.
    class InvokeAnthropic < Anthropic
      include InvokeAnthropic::Chat
      include InvokeAnthropic::Streaming

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
