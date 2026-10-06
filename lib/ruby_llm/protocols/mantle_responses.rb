# frozen_string_literal: true

module RubyLLM
  module Protocols
    # AWS Bedrock's bedrock-mantle endpoint, speaking the OpenAI Responses API.
    # Reachable only via bedrock-mantle (not bedrock-runtime), and only for models
    # that physically cannot serve Converse (the GPT-5.x and GPT-6.x frontier
    # families). Talks to the provider's mantle connection instead of the default
    # bedrock-runtime connection, and SigV4-signs every request against the
    # "bedrock-mantle" service namespace instead of "bedrock".
    class MantleResponses < Responses
      # Frontier openai.gpt-5.x and openai.gpt-6.x models are served at /openai/v1/responses;
      # every other mantle model (e.g. openai.gpt-oss-*) is served at /v1/responses. See the
      # AWS Bedrock model cards for GPT-5.6 and GPT-6.
      #
      # Kept in sync with Providers::Bedrock::MANTLE_ONLY_MODEL_PATTERN — that pattern picks
      # mantle vs Converse, this one picks the /openai/v1 vs /v1 mantle path. They
      # coincide today but are distinct concepts; update both when a new frontier family lands.
      FRONTIER_GPT_PATTERN = /\Aopenai\.gpt-[56]/

      def initialize(provider, model = nil)
        super
        @connection = provider.mantle_connection
      end

      # The '/v1/responses' branch is forward-looking: today Providers::Bedrock#protocol_for only
      # ever routes /\Aopenai\.gpt-[56]/ ids here, so this branch is unreachable in production (e.g.
      # openai.gpt-oss-* still routes to Converse). It exists so this protocol is ready if a
      # future mantle-only, non-frontier model needs it, without another round of plumbing.
      def completion_url
        FRONTIER_GPT_PATTERN.match?(@model.id) ? '/openai/v1/responses' : '/v1/responses'
      end

      private

      def sync_response(payload, additional_headers = {})
        super(payload, additional_headers.merge(mantle_signature_headers(completion_url, payload)))
      end

      def stream_response(payload, additional_headers = {}, &)
        super(payload, additional_headers.merge(mantle_signature_headers(completion_url, payload)), &)
      end

      def compaction_response(payload, additional_headers = {})
        super(payload, additional_headers.merge(mantle_signature_headers(compaction_url, payload)))
      end

      def mantle_signature_headers(path, payload)
        body = JSON.generate(payload)
        @provider.sign_headers(
          'POST', path, body,
          base_url: @provider.mantle_api_base,
          service: Providers::Bedrock::MANTLE_SIGNING_SERVICE,
          region: @provider.mantle_region
        )
      end
    end
  end
end
