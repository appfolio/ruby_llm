# frozen_string_literal: true

module RubyLLM
  module Providers
    # AWS Bedrock integration.
    class Bedrock < Provider
      include Bedrock::Auth
      include Bedrock::Models

      protocol :converse, Protocols::Converse, batches: Protocols::Converse::Batches
      protocol :mantle_responses, Protocols::MantleResponses
      files Bedrock::Files

      # SigV4 requests to bedrock-mantle sign against this service namespace, not "bedrock" —
      # AWS models bedrock-mantle as a separate service (see the AmazonBedrockMantleFullAccess
      # managed policy and the bedrock-mantle:CreateInference IAM action). Named here so it is a
      # one-line change if this assumption turns out to be wrong.
      MANTLE_SIGNING_SERVICE = 'bedrock-mantle'

      # openai.gpt-5.x ids are only reachable on bedrock-mantle, not Converse.
      # Deliberately narrower than /\Aopenai\./ — openai.gpt-oss-* models ARE served by
      # bedrock-runtime and must keep routing to Converse.
      MANTLE_ONLY_MODEL_PATTERN = /\Aopenai\.gpt-5/

      # Converse-specific params that Mantle's Responses API would reject outright.
      CONVERSE_ONLY_PARAMS = %i[top_k additionalModelRequestFields].freeze

      def api_base
        @config.bedrock_api_base || "https://bedrock-runtime.#{bedrock_region}.amazonaws.com"
      end

      def control_api_base
        @config.bedrock_api_base || "https://bedrock.#{bedrock_region}.amazonaws.com"
      end

      def mantle_api_base
        @config.bedrock_mantle_api_base || "https://bedrock-mantle.#{mantle_region}.api.aws"
      end

      def mantle_connection
        @mantle_connection ||= Connection.new(self, @config, base_url: mantle_api_base)
      end

      def mantle_region
        @config.bedrock_mantle_region || bedrock_region
      end

      def headers
        {}
      end

      def complete(messages, model:, params: {}, **rest, &)
        params = mantle_only_model?(model) ? strip_converse_only_params(params) : normalize_params(params, model:)
        # Bare `super` forwards current bindings, so it picks up the reassigned `params` above.
        super
      end

      def protocol_for(model, **)
        return fetch_protocol(:mantle_responses) if mantle_only_model?(model)

        fetch_protocol(:converse)
      end

      def parse_error(response)
        return if response.body.nil? || response.body.empty?

        body = try_parse_json(response.body)
        return body if body.is_a?(String)

        extract_error_message(body) || super
      end

      def list_models
        response = signed_get(models_api_base, models_url)
        parse_list_models_response(response, slug, capabilities)
      end

      class << self
        def configuration_options
          %i[
            bedrock_api_key
            bedrock_secret_key
            bedrock_region
            bedrock_session_token
            bedrock_credential_provider
            bedrock_api_base
            bedrock_mantle_api_base
            bedrock_mantle_region
            bedrock_batch_s3_uri
            bedrock_batch_role_arn
          ]
        end

        def configuration_requirements
          %i[bedrock_region]
        end

        def configured?(config)
          !!(config.bedrock_region && credentials_configured?(config))
        end

        def credentials_configured?(config)
          return credential_provider?(config) if config.bedrock_credential_provider

          !!(config.bedrock_api_key && config.bedrock_secret_key)
        end

        private

        def credential_provider?(config)
          config.bedrock_credential_provider&.respond_to?(:credentials)
        end
      end

      def ensure_configured!
        return if configured?

        missing = []
        missing << :bedrock_region unless @config.bedrock_region
        missing << bedrock_credentials_requirement unless self.class.credentials_configured?(@config)

        raise ConfigurationError, "Missing configuration for Bedrock: #{missing.join(', ')}"
      end

      private

      # Bedrock errors are shaped like {"message" => "..."} or {"__type" => "..."};
      # mantle (OpenAI Responses) errors are shaped like {"error" => {"message" => "..."}}.
      def extract_error_message(body)
        nested_message = body['error'].is_a?(Hash) ? body.dig('error', 'message') : nil
        nested_message || body['message'] || body['Message'] || body['error'] || body['__type']
      end

      def bedrock_region
        @config.bedrock_region
      end

      # openai.gpt-5.x ids cannot serve Converse; routing here is automatic.
      #
      # Kept in sync with Protocols::MantleResponses::FRONTIER_GPT5_PATTERN — that pattern picks
      # the /openai/v1 vs /v1 mantle path, this one picks mantle vs Converse. They
      # coincide today but are distinct concepts; update both when a new frontier family lands.
      def mantle_only_model?(model)
        MANTLE_ONLY_MODEL_PATTERN.match?(model.id.to_s)
      end

      def bedrock_credentials_requirement
        if @config.bedrock_credential_provider
          'bedrock_credential_provider responding to #credentials'
        else
          'bedrock_credential_provider or bedrock_api_key + bedrock_secret_key'
        end
      end

      def normalize_params(params, model:)
        normalized = RubyLLM::Utils.deep_symbolize_keys(params || {})
        additional_fields = normalized[:additionalModelRequestFields] || {}

        top_k = normalized.delete(:top_k)
        if !top_k.nil? && model_supports_top_k?(model)
          additional_fields = RubyLLM::Utils.deep_merge(additional_fields, { top_k: top_k })
        end

        normalized[:additionalModelRequestFields] = additional_fields unless additional_fields.empty?
        normalized
      end

      def model_supports_top_k?(model)
        Protocols::Converse.reasoning_embedded?(model)
      end

      # Mantle speaks the OpenAI Responses API, not Converse, so Converse-only params would
      # otherwise be forwarded raw and rejected with an opaque 400 from the mantle endpoint.
      def strip_converse_only_params(params)
        normalized = RubyLLM::Utils.deep_symbolize_keys(params || {})
        offending = CONVERSE_ONLY_PARAMS & normalized.keys
        return normalized if offending.empty?

        raise ArgumentError,
              "#{offending.join(', ')} are Converse-only params and are not supported on " \
              'bedrock-mantle (Responses API)'
      end
    end
  end
end
