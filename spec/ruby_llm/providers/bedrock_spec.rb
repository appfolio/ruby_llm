# frozen_string_literal: true

require 'spec_helper'

RSpec.describe RubyLLM::Providers::Bedrock do
  let(:credentials_class) { Struct.new(:access_key_id, :secret_access_key, :session_token, keyword_init: true) }
  let(:credential_provider_class) { Struct.new(:credentials, keyword_init: true) }

  def bedrock_config(region: 'us-east-1', api_key: nil, secret_key: nil, session_token: nil, credential_provider: nil)
    RubyLLM::Configuration.new.tap do |config|
      config.bedrock_region = region
      config.bedrock_api_key = api_key
      config.bedrock_secret_key = secret_key
      config.bedrock_session_token = session_token
      config.bedrock_credential_provider = credential_provider
    end
  end

  def credentials(access_key_id: 'provider-key', secret_access_key: 'provider-secret', session_token: 'provider-token')
    credentials_class.new(access_key_id:, secret_access_key:, session_token:)
  end

  def credential_provider(credentials = self.credentials)
    credential_provider_class.new(credentials:)
  end

  describe '.configuration_options' do
    it 'registers credential providers as a Bedrock option' do
      expect(RubyLLM::Configuration.options).to include(:bedrock_credential_provider)
    end

    it 'registers the mantle base URL and region as optional Bedrock options' do
      expect(RubyLLM::Configuration.options).to include(:bedrock_mantle_api_base, :bedrock_mantle_region)
    end

    it 'does not require the mantle options' do
      expect(described_class.configuration_requirements).not_to include(:bedrock_mantle_api_base,
                                                                        :bedrock_mantle_region)
    end
  end

  describe '#mantle_api_base and #mantle_connection' do
    it 'defaults the mantle base URL from bedrock_region' do
      provider = described_class.new(bedrock_config(region: 'us-west-2', api_key: 'k', secret_key: 's'))

      expect(provider.mantle_api_base).to eq('https://bedrock-mantle.us-west-2.api.aws')
    end

    it 'falls back to bedrock_region when bedrock_mantle_region is unset' do
      config = bedrock_config(region: 'us-east-1', api_key: 'k', secret_key: 's')
      provider = described_class.new(config)

      expect(provider.mantle_api_base).to eq('https://bedrock-mantle.us-east-1.api.aws')
    end

    it 'prefers bedrock_mantle_region over bedrock_region when set' do
      config = bedrock_config(region: 'us-east-1', api_key: 'k', secret_key: 's')
      config.bedrock_mantle_region = 'us-west-2'
      provider = described_class.new(config)

      expect(provider.mantle_api_base).to eq('https://bedrock-mantle.us-west-2.api.aws')
    end

    it 'is overridable via bedrock_mantle_api_base' do
      config = bedrock_config(region: 'us-west-2', api_key: 'k', secret_key: 's')
      config.bedrock_mantle_api_base = 'https://custom.mantle.example.com'
      provider = described_class.new(config)

      expect(provider.mantle_api_base).to eq('https://custom.mantle.example.com')
    end

    it 'builds a connection bound to the mantle base URL' do
      provider = described_class.new(bedrock_config(region: 'us-west-2', api_key: 'k', secret_key: 's'))

      expect(provider.mantle_connection.connection.url_prefix.to_s).to eq('https://bedrock-mantle.us-west-2.api.aws/')
    end
  end

  describe '.configured?' do
    it 'accepts static credentials with a region' do
      config = bedrock_config(api_key: 'static-key', secret_key: 'static-secret')

      expect(described_class.configured?(config)).to be(true)
    end

    it 'accepts a credential provider with a region' do
      config = bedrock_config(credential_provider: credential_provider)

      expect(described_class.configured?(config)).to be(true)
    end

    it 'rejects a region without credentials' do
      config = bedrock_config

      expect(described_class.configured?(config)).to be(false)
    end

    it 'rejects credentials without a region' do
      config = bedrock_config(region: nil, credential_provider: credential_provider)

      expect(described_class.configured?(config)).to be(false)
    end

    it 'rejects an invalid credential provider instead of falling back to static keys' do
      config = bedrock_config(
        api_key: 'static-key',
        secret_key: 'static-secret',
        credential_provider: Object.new
      )

      expect(described_class.configured?(config)).to be(false)
    end
  end

  describe '#initialize' do
    it 'explains the alternative credential shapes' do
      expect { described_class.new(bedrock_config) }
        .to raise_error(RubyLLM::ConfigurationError, /bedrock_credential_provider or bedrock_api_key/)
    end

    it 'explains an invalid credential provider' do
      config = bedrock_config(
        api_key: 'static-key',
        secret_key: 'static-secret',
        credential_provider: Object.new
      )

      expect { described_class.new(config) }
        .to raise_error(RubyLLM::ConfigurationError, /bedrock_credential_provider responding to #credentials/)
    end
  end

  describe '#sign_headers' do
    it 'signs with static credentials' do
      provider = described_class.new(
        bedrock_config(api_key: 'static-key', secret_key: 'static-secret', session_token: 'static-token')
      )

      headers = provider.sign_headers('POST', '/model/anthropic.claude-haiku/converse', '{}')

      expect(headers['Authorization']).to include('Credential=static-key/')
      expect(headers['X-Amz-Security-Token']).to eq('static-token')
    end

    it 'signs with a credential provider instead of configured static credentials' do
      provider = described_class.new(
        bedrock_config(
          api_key: 'static-key',
          secret_key: 'static-secret',
          session_token: 'static-token',
          credential_provider: credential_provider
        )
      )

      headers = provider.sign_headers('POST', '/model/anthropic.claude-haiku/converse', '{}')

      expect(headers['Authorization']).to include('Credential=provider-key/')
      expect(headers['X-Amz-Security-Token']).to eq('provider-token')
    end

    it 'defaults the signing service to "bedrock" for existing callers' do
      provider = described_class.new(bedrock_config(api_key: 'static-key', secret_key: 'static-secret'))

      headers = provider.sign_headers('POST', '/model/anthropic.claude-haiku/converse', '{}')

      expect(headers['Authorization']).to include("/#{provider.send(:bedrock_region)}/bedrock/aws4_request")
    end

    it 'signs against an explicit service name when given' do
      provider = described_class.new(bedrock_config(api_key: 'static-key', secret_key: 'static-secret'))

      headers = provider.sign_headers('POST', '/openai/v1/responses', '{}', service: 'bedrock-mantle')

      expect(headers['Authorization']).to include("/#{provider.send(:bedrock_region)}/bedrock-mantle/aws4_request")
    end

    it 'defaults the signing region to bedrock_region for existing callers' do
      provider = described_class.new(
        bedrock_config(region: 'us-west-2', api_key: 'static-key', secret_key: 'static-secret')
      )

      headers = provider.sign_headers('POST', '/model/anthropic.claude-haiku/converse', '{}')

      expect(headers['Authorization']).to include('/us-west-2/bedrock/aws4_request')
    end

    it 'signs against an explicit region when given, independent of bedrock_region' do
      provider = described_class.new(
        bedrock_config(region: 'us-west-2', api_key: 'static-key', secret_key: 'static-secret')
      )

      headers = provider.sign_headers('POST', '/openai/v1/responses', '{}', service: 'bedrock-mantle',
                                                                            region: 'us-east-2')

      expect(headers['Authorization']).to include('/us-east-2/bedrock-mantle/aws4_request')
    end
  end

  describe '#parse_error' do
    let(:provider) { described_class.new(bedrock_config(api_key: 'k', secret_key: 's')) }

    def response_double(body)
      instance_double(Faraday::Response, body: body)
    end

    it 'extracts the message from a Bedrock-shaped error body' do
      response = response_double('message' => 'model not found')

      expect(provider.parse_error(response)).to eq('model not found')
    end

    it 'extracts the message from a Bedrock-shaped __type body' do
      response = response_double('__type' => 'ValidationException')

      expect(provider.parse_error(response)).to eq('ValidationException')
    end

    it 'extracts the nested message from an OpenAI-shaped (mantle) error body' do
      response = response_double('error' => { 'message' => 'invalid request', 'type' => 'invalid_request_error' })

      expect(provider.parse_error(response)).to eq('invalid request')
    end
  end

  describe '#complete params normalization' do
    let(:provider) { described_class.new(bedrock_config(api_key: 'k', secret_key: 's')) }

    def model_double(id, metadata: {})
      instance_double(RubyLLM::Model::Info, id: id, max_tokens: 4096, metadata: metadata, provider: 'bedrock')
    end

    it 'injects additionalModelRequestFields (top_k) for Converse-routed reasoning-embedded models' do
      model = model_double(
        'anthropic.claude-haiku-4-5-20251001-v1:0',
        metadata: { converse: { reasoningSupported: { embedded: true } } }
      )
      protocol = instance_double(RubyLLM::Protocols::Converse)
      allow(RubyLLM::Protocols::Converse).to receive(:new).and_return(protocol)
      allow(protocol).to receive(:complete)

      provider.complete([], model: model, tools: {}, temperature: nil, params: { top_k: 5 })

      expect(protocol).to have_received(:complete).with(
        [], hash_including(params: { additionalModelRequestFields: { top_k: 5 } }), any_args
      )
    end

    it 'does not run normalize_params (Converse-specific) for mantle-routed models, forwarding params as-is' do
      model = model_double('openai.gpt-5.6-sol')
      protocol = instance_double(RubyLLM::Protocols::MantleResponses)
      allow(RubyLLM::Protocols::MantleResponses).to receive(:new).and_return(protocol)
      allow(protocol).to receive(:complete)

      provider.complete([], model: model, tools: {}, temperature: nil, params: { top_k: 5 })

      expect(protocol).to have_received(:complete).with([], hash_including(params: { top_k: 5 }), any_args)
    end
  end

  describe '#protocol_for / #invoke_model? / #anthropic_model?' do # rubocop:disable RSpec/MultipleMemoizedHelpers
    def build_bedrock(use_invoke_model: false)
      config = bedrock_config(api_key: 'k', secret_key: 's')
      config.bedrock_use_invoke_model = use_invoke_model
      described_class.new(config)
    end

    def model_double(id, metadata: {}, provider: 'bedrock')
      instance_double(
        RubyLLM::Model::Info,
        id: id,
        max_tokens: 4096,
        metadata: metadata,
        provider: provider
      )
    end

    let(:haiku_id)  { 'anthropic.claude-haiku-4-5-20251001-v1:0' }
    let(:nova_id)   { 'amazon.nova-lite-v1:0' }
    let(:llama_id)  { 'meta.llama3-8b-instruct-v1:0' }
    let(:arn_id)    { 'arn:aws:bedrock:us-west-2:123456789012:application-inference-profile/p' }
    let(:us_sonnet_id) { 'us.anthropic.claude-sonnet-5' }
    let(:eu_haiku_id)  { 'eu.anthropic.claude-haiku-4-5-20251001-v1:0' }
    let(:us_nova_id)   { 'us.amazon.nova-pro-v1:0' }

    context 'with bedrock_use_invoke_model: false (default)' do # rubocop:disable RSpec/MultipleMemoizedHelpers
      let(:provider) { build_bedrock(use_invoke_model: false) }

      it 'routes all models to Converse' do
        expect(provider.protocol_for(model_double(haiku_id))).to be(RubyLLM::Protocols::Converse)
        expect(provider.protocol_for(model_double(nova_id))).to be(RubyLLM::Protocols::Converse)
      end
    end

    context 'with bedrock_use_invoke_model: nil' do # rubocop:disable RSpec/MultipleMemoizedHelpers
      let(:provider) { build_bedrock(use_invoke_model: nil) }

      it 'routes all models to Converse' do
        expect(provider.protocol_for(model_double(haiku_id))).to be(RubyLLM::Protocols::Converse)
      end
    end

    context 'with bedrock_use_invoke_model: true' do # rubocop:disable RSpec/MultipleMemoizedHelpers
      let(:provider) { build_bedrock(use_invoke_model: true) }

      it 'routes Anthropic models (anthropic.* prefix) to BedrockInvokeModel' do
        expect(provider.protocol_for(model_double(haiku_id))).to be(RubyLLM::Protocols::BedrockInvokeModel)
      end

      it 'routes non-Anthropic models to Converse regardless of flag' do
        expect(provider.protocol_for(model_double(nova_id))).to be(RubyLLM::Protocols::Converse)
        expect(provider.protocol_for(model_double(llama_id))).to be(RubyLLM::Protocols::Converse)
      end

      it 'routes ARN ids with Anthropic provider_name to BedrockInvokeModel' do
        model = model_double(arn_id, metadata: { provider_name: 'Anthropic' })
        expect(provider.protocol_for(model)).to be(RubyLLM::Protocols::BedrockInvokeModel)
      end

      it 'routes ARN ids without provider_name to Converse with a warning' do
        allow(RubyLLM.logger).to receive(:warn)
        expect(provider.protocol_for(model_double(arn_id))).to be(RubyLLM::Protocols::Converse)
        expect(RubyLLM.logger).to have_received(:warn).with(/cannot verify.*Anthropic-backed/)
      end

      it 'routes cross-region inference profile ids (us.anthropic.*) to BedrockInvokeModel' do
        model = model_double(us_sonnet_id, provider: 'bedrock')
        expect(provider.protocol_for(model)).to be(RubyLLM::Protocols::BedrockInvokeModel)
      end

      it 'routes cross-region inference profile ids (eu.anthropic.*) to BedrockInvokeModel' do
        model = model_double(eu_haiku_id, provider: 'bedrock')
        expect(provider.protocol_for(model)).to be(RubyLLM::Protocols::BedrockInvokeModel)
      end

      it 'routes cross-region non-Anthropic profile ids (us.amazon.*) to Converse' do
        model = model_double(us_nova_id, provider: 'bedrock')
        expect(provider.protocol_for(model)).to be(RubyLLM::Protocols::Converse)
      end
    end

    context 'with Array selector' do # rubocop:disable RSpec/MultipleMemoizedHelpers
      let(:sonnet_id) { 'anthropic.claude-sonnet-4-6-20250514-v1:0' }
      let(:provider)  { build_bedrock(use_invoke_model: [sonnet_id]) }

      it 'routes only listed model ids to BedrockInvokeModel' do
        expect(provider.protocol_for(model_double(sonnet_id))).to be(RubyLLM::Protocols::BedrockInvokeModel)
      end

      it 'routes unlisted Anthropic models to Converse' do
        expect(provider.protocol_for(model_double(haiku_id))).to be(RubyLLM::Protocols::Converse)
      end
    end

    context 'with Array selector containing a cross-region inference profile id' do # rubocop:disable RSpec/MultipleMemoizedHelpers
      let(:provider) { build_bedrock(use_invoke_model: [us_sonnet_id]) }

      it 'routes the listed cross-region model id to BedrockInvokeModel' do
        model = model_double(us_sonnet_id, provider: 'bedrock')
        expect(provider.protocol_for(model)).to be(RubyLLM::Protocols::BedrockInvokeModel)
      end

      it 'routes an unlisted cross-region model id to Converse' do
        model = model_double(eu_haiku_id, provider: 'bedrock')
        expect(provider.protocol_for(model)).to be(RubyLLM::Protocols::Converse)
      end
    end

    context 'with Proc/lambda selector' do # rubocop:disable RSpec/MultipleMemoizedHelpers
      let(:sonnet_id) { 'anthropic.claude-sonnet-4-6-20250514-v1:0' }
      let(:selector)  { ->(m) { m.id == sonnet_id } }
      let(:provider)  { build_bedrock(use_invoke_model: selector) }

      it 'routes models where the callable returns true to BedrockInvokeModel' do
        expect(provider.protocol_for(model_double(sonnet_id))).to be(RubyLLM::Protocols::BedrockInvokeModel)
      end

      it 'routes models where the callable returns false to Converse' do
        expect(provider.protocol_for(model_double(haiku_id))).to be(RubyLLM::Protocols::Converse)
      end
    end

    context 'with an unverifiable ARN id (no provider_name metadata)' do # rubocop:disable RSpec/MultipleMemoizedHelpers
      before { allow(RubyLLM.logger).to receive(:warn) }

      it 'routes to BedrockInvokeModel when an Array selector lists the ARN' do
        provider = build_bedrock(use_invoke_model: [arn_id])
        expect(provider.protocol_for(model_double(arn_id))).to be(RubyLLM::Protocols::BedrockInvokeModel)
      end

      it 'routes to BedrockInvokeModel when a Proc selector returns true' do
        provider = build_bedrock(use_invoke_model: ->(_m) { true })
        expect(provider.protocol_for(model_double(arn_id))).to be(RubyLLM::Protocols::BedrockInvokeModel)
      end

      it 'routes to Converse when a Proc selector returns false' do
        provider = build_bedrock(use_invoke_model: ->(_m) { false })
        expect(provider.protocol_for(model_double(arn_id))).to be(RubyLLM::Protocols::Converse)
      end

      it 'still requires positive verification under the blanket true selector' do
        provider = build_bedrock(use_invoke_model: true)
        expect(provider.protocol_for(model_double(arn_id))).to be(RubyLLM::Protocols::Converse)
        expect(RubyLLM.logger).to have_received(:warn).with(/cannot verify.*Anthropic-backed/)
      end
    end

    context 'with provably non-Anthropic ids under explicit selectors' do # rubocop:disable RSpec/MultipleMemoizedHelpers
      it 'never routes a bare vendor id, even when an Array selector lists it' do
        provider = build_bedrock(use_invoke_model: [nova_id])
        expect(provider.protocol_for(model_double(nova_id))).to be(RubyLLM::Protocols::Converse)
      end

      it 'never routes a cross-region vendor id, even when a Proc selector returns true' do
        provider = build_bedrock(use_invoke_model: ->(_m) { true })
        expect(provider.protocol_for(model_double(us_nova_id))).to be(RubyLLM::Protocols::Converse)
        expect(provider.protocol_for(model_double(llama_id))).to be(RubyLLM::Protocols::Converse)
      end
    end

    context 'with GPT-5.6 (mantle-only) and gpt-oss (bedrock-runtime) model ids' do # rubocop:disable RSpec/MultipleMemoizedHelpers
      let(:provider) { build_bedrock(use_invoke_model: true) }

      it 'routes openai.gpt-5.6-sol/-terra/-luna to MantleResponses regardless of bedrock_use_invoke_model' do
        %w[openai.gpt-5.6-sol openai.gpt-5.6-terra openai.gpt-5.6-luna].each do |id|
          expect(provider.protocol_for(model_double(id))).to be(RubyLLM::Protocols::MantleResponses)
        end
      end

      it 'keeps routing openai.gpt-oss-120b and openai.gpt-oss-20b to Converse' do
        expect(provider.protocol_for(model_double('openai.gpt-oss-120b'))).to be(RubyLLM::Protocols::Converse)
        expect(provider.protocol_for(model_double('openai.gpt-oss-20b'))).to be(RubyLLM::Protocols::Converse)
      end

      it 'keeps routing anthropic.* ids to BedrockInvokeModel exactly as before' do
        expect(provider.protocol_for(model_double(haiku_id))).to be(RubyLLM::Protocols::BedrockInvokeModel)
      end
    end

    describe '#mantle_only_model?' do # rubocop:disable RSpec/MultipleMemoizedHelpers
      let(:bedrock) { build_bedrock }

      it 'matches openai.gpt-5.x ids narrowly, not the broad openai. prefix' do
        expect(bedrock.send(:mantle_only_model?, model_double('openai.gpt-5.6-sol'))).to be(true)
        expect(bedrock.send(:mantle_only_model?, model_double('openai.gpt-oss-120b'))).to be(false)
        expect(bedrock.send(:mantle_only_model?, model_double('openai.gpt-5.5'))).to be(true)
      end
    end

    describe 'anthropic_model? directly' do # rubocop:disable RSpec/MultipleMemoizedHelpers
      let(:bedrock) { build_bedrock }

      it 'returns true for anthropic.* model ids' do
        expect(bedrock.send(:anthropic_model?, model_double('anthropic.claude-3-haiku'))).to be(true)
      end

      it 'returns false for amazon.* (Nova) model ids' do
        expect(bedrock.send(:anthropic_model?, model_double('amazon.nova-lite-v1:0'))).to be(false)
      end

      it 'returns false for meta.* (Llama) model ids' do
        expect(bedrock.send(:anthropic_model?, model_double('meta.llama3-8b-instruct-v1:0'))).to be(false)
      end

      it 'returns false for other NON_ANTHROPIC_PREFIXES vendors' do
        %w[ai21. cohere. mistral. writer. stability.].each do |prefix|
          expect(bedrock.send(:anthropic_model?, model_double("#{prefix}some-model"))).to be(false)
        end
      end

      it 'returns true for cross-region inference profile ids regardless of geo prefix' do
        %w[us. eu. apac. global. jp. au. us-gov.].each do |geo|
          id = "#{geo}anthropic.claude-sonnet-5"
          expect(bedrock.send(:anthropic_model?, model_double(id, provider: 'bedrock'))).to be(true)
        end
      end

      it 'returns false for cross-region profile ids of non-Anthropic vendors' do
        expect(bedrock.send(:anthropic_model?, model_double('us.amazon.nova-pro-v1:0', provider: 'bedrock')))
          .to be(false)
        expect(bedrock.send(:anthropic_model?,
                            model_double('us.meta.llama3-1-405b-instruct-v1:0', provider: 'bedrock')))
          .to be(false)
      end

      it 'returns true for ARN ids whose metadata shows Anthropic as provider' do
        model = model_double(arn_id, metadata: { provider_name: 'Anthropic' })
        expect(bedrock.send(:anthropic_model?, model)).to be(true)
      end

      it 'returns false and logs a warning for ARN ids without provider_name metadata' do
        allow(RubyLLM.logger).to receive(:warn)
        expect(bedrock.send(:anthropic_model?, model_double(arn_id))).to be(false)
        expect(RubyLLM.logger).to have_received(:warn).with(/cannot verify.*Anthropic-backed/)
      end

      it 'returns true for models whose provider field is "anthropic" (fallback)' do
        model = model_double('unknown-model', provider: 'anthropic')
        expect(bedrock.send(:anthropic_model?, model)).to be(true)
      end

      it 'returns false for models with non-anthropic provider field and unknown prefix' do
        model = model_double('unknown-model', provider: 'bedrock')
        expect(bedrock.send(:anthropic_model?, model)).to be(false)
      end
    end
  end

  describe 'model id path encoding' do
    # completion_url/stream_url read only @model, and canonical_uri is a pure path
    # transform, so allocate uninitialized instances to keep these tests focused and
    # credential/connection-free.
    let(:converse) { RubyLLM::Protocols::Converse.allocate }
    let(:arn) { 'arn:aws:bedrock:us-west-2:123:application-inference-profile/p' }

    def with_model(id)
      converse.instance_variable_set(:@model, instance_double(RubyLLM::Model::Info, id: id))
    end

    it 'keeps an application inference profile ARN as a single path segment in the converse URL' do
      with_model(arn)
      # The ARN's internal "/" is percent-encoded so it is not parsed as a path separator
      # (which would truncate the modelId to ".../application-inference-profile").
      expect(converse.send(:completion_url)).to eq(
        '/model/arn:aws:bedrock:us-west-2:123:application-inference-profile%2Fp/converse'
      )
    end

    it 'encodes the ARN for the converse-stream URL too' do
      with_model(arn)
      expect(converse.send(:stream_url)).to eq(
        '/model/arn:aws:bedrock:us-west-2:123:application-inference-profile%2Fp/converse-stream'
      )
    end

    it 'leaves ordinary model ids (including a ":" version suffix) unchanged' do
      with_model('us.anthropic.claude-sonnet-4-5-20250929-v1:0')
      expect(converse.send(:completion_url)).to eq(
        '/model/us.anthropic.claude-sonnet-4-5-20250929-v1:0/converse'
      )
    end

    it 'signs the ARN as one segment (SigV4 canonical path double-encodes "/", not truncates)' do
      with_model(arn)
      path = URI.parse(converse.send(:completion_url)).path
      # canonical_uri re-encodes each segment, turning the already-encoded "%2F" into
      # "%252F" — so the profile id stays inside the modelId segment rather than becoming
      # its own path segment, keeping the signed path consistent with the sent path.
      expect(described_class.allocate.send(:canonical_uri, path)).to eq(
        '/model/arn%3Aaws%3Abedrock%3Aus-west-2%3A123%3Aapplication-inference-profile%252Fp/converse'
      )
    end
  end
end
