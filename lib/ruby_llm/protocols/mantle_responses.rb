# frozen_string_literal: true

module RubyLLM
  module Protocols
    # AWS Bedrock's bedrock-mantle endpoint, speaking the OpenAI Responses API.
    # Reachable only via bedrock-mantle (not bedrock-runtime), and only for models
    # that physically cannot serve Converse or InvokeModel (the GPT-5.x frontier
    # family). Talks to the provider's mantle connection instead of the default
    # bedrock-runtime connection, and SigV4-signs every request against the
    # "bedrock-mantle" service namespace instead of "bedrock".
    class MantleResponses < Responses
      # Frontier openai.gpt-5.x models are served at /openai/v1/responses; every other
      # mantle model (e.g. openai.gpt-oss-*) is served at /v1/responses. See the
      # AWS Bedrock model cards for GPT-5.6.
      FRONTIER_GPT5_PATTERN = /\Aopenai\.gpt-5/

      def initialize(provider, model = nil)
        super
        @connection = provider.mantle_connection
      end

      def completion_url
        FRONTIER_GPT5_PATTERN.match?(@model.id) ? '/openai/v1/responses' : '/v1/responses'
      end

      private

      def sync_response(payload, additional_headers = {})
        super(payload, additional_headers.merge(mantle_signature_headers(payload)))
      end

      def stream_response(payload, additional_headers = {}, &)
        super(payload, additional_headers.merge(mantle_signature_headers(payload)), &)
      end

      def mantle_signature_headers(payload)
        body = JSON.generate(payload)
        @provider.sign_headers(
          'POST', completion_url, body,
          base_url: @provider.mantle_api_base,
          service: Providers::Bedrock::MANTLE_SIGNING_SERVICE,
          region: @provider.mantle_region
        )
      end
    end
  end
end
