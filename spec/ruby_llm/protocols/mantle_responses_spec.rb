# frozen_string_literal: true

require 'spec_helper'

RSpec.describe RubyLLM::Protocols::MantleResponses do
  let(:config) do
    RubyLLM::Configuration.new.tap do |c|
      c.bedrock_region = 'us-west-2'
      c.bedrock_api_key = 'static-key'
      c.bedrock_secret_key = 'static-secret'
    end
  end
  let(:provider) { RubyLLM::Providers::Bedrock.new(config) }

  def model_info(id)
    RubyLLM::Model::Info.new(
      id: id, name: id, provider: 'bedrock', family: 'gpt', created_at: nil,
      context_window: 1, max_output_tokens: 1,
      modalities: { input: [], output: [] }, capabilities: [], pricing: {}, metadata: {}
    )
  end

  describe '#completion_url' do
    it 'uses /openai/v1/responses for frontier openai.gpt-5.x ids' do
      %w[openai.gpt-5.6-sol openai.gpt-5.6-terra openai.gpt-5.6-luna openai.gpt-5.5].each do |id|
        protocol = described_class.new(provider, model_info(id))
        expect(protocol.completion_url).to eq('/openai/v1/responses')
      end
    end

    it 'uses /v1/responses for non-frontier mantle ids' do
      protocol = described_class.new(provider, model_info('openai.gpt-oss-120b'))
      expect(protocol.completion_url).to eq('/v1/responses')
    end
  end

  describe '#initialize' do
    it 'binds to the provider mantle connection instead of the default bedrock-runtime connection' do
      protocol = described_class.new(provider, model_info('openai.gpt-5.6-sol'))

      expect(protocol.connection).to be(provider.mantle_connection)
      expect(protocol.connection.connection.url_prefix.to_s).to eq('https://bedrock-mantle.us-west-2.api.aws/')
    end
  end

  describe 'request signing' do
    def response_body
      { output: [], usage: {} }.to_json
    end

    it 'signs the sync request against bedrock-mantle at the frontier path' do
      stub = stub_request(:post, 'https://bedrock-mantle.us-west-2.api.aws/openai/v1/responses')
             .with { |req| req.headers['Authorization']&.include?('us-west-2/bedrock-mantle/aws4_request') }
             .to_return(status: 200, body: response_body, headers: { 'Content-Type' => 'application/json' })

      protocol = described_class.new(provider, model_info('openai.gpt-5.6-sol'))
      protocol.complete([RubyLLM::Message.new(role: :user, content: 'hi')], tools: {}, temperature: nil)

      expect(stub).to have_been_requested
    end

    it 'signs the sync request against bedrock-mantle at the non-frontier path' do
      stub = stub_request(:post, 'https://bedrock-mantle.us-west-2.api.aws/v1/responses')
             .with { |req| req.headers['Authorization']&.include?('bedrock-mantle/aws4_request') }
             .to_return(status: 200, body: response_body, headers: { 'Content-Type' => 'application/json' })

      protocol = described_class.new(provider, model_info('openai.gpt-oss-120b'))
      protocol.complete([RubyLLM::Message.new(role: :user, content: 'hi')], tools: {}, temperature: nil)

      expect(stub).to have_been_requested
    end

    it 'signs the streaming request against bedrock-mantle too' do
      sse = "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{}}}\n\n"
      stub = stub_request(:post, 'https://bedrock-mantle.us-west-2.api.aws/openai/v1/responses')
             .with { |req| req.headers['Authorization']&.include?('bedrock-mantle/aws4_request') }
             .to_return(status: 200, body: sse, headers: { 'Content-Type' => 'text/event-stream' })

      protocol = described_class.new(provider, model_info('openai.gpt-5.6-terra'))
      chunks = []
      protocol.complete([RubyLLM::Message.new(role: :user, content: 'hi')], tools: {}, temperature: nil) do |chunk|
        chunks << chunk
      end

      expect(stub).to have_been_requested
      expect(chunks).not_to be_empty
    end

    it 'signs against bedrock_mantle_region, not bedrock_region, when they differ' do
      config.bedrock_mantle_region = 'us-east-2'

      stub = stub_request(:post, 'https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses')
             .with { |req| req.headers['Authorization']&.include?('us-east-2/bedrock-mantle/aws4_request') }
             .to_return(status: 200, body: response_body, headers: { 'Content-Type' => 'application/json' })

      protocol = described_class.new(provider, model_info('openai.gpt-5.6-sol'))
      protocol.complete([RubyLLM::Message.new(role: :user, content: 'hi')], tools: {}, temperature: nil)

      expect(stub).to have_been_requested
    end

    it 'signs the exact bytes Faraday sends, not a re-serialized copy' do
      stub = stub_request(:post, 'https://bedrock-mantle.us-west-2.api.aws/openai/v1/responses')
             .to_return(status: 200, body: response_body, headers: { 'Content-Type' => 'application/json' })

      protocol = described_class.new(provider, model_info('openai.gpt-5.6-sol'))
      protocol.complete([RubyLLM::Message.new(role: :user, content: 'hi')], tools: {}, temperature: nil)

      expect(stub).to have_been_requested

      sent_request = WebMock::RequestRegistry.instance.requested_signatures.hash.keys.find do |req|
        req.uri.host == 'bedrock-mantle.us-west-2.api.aws' && req.uri.path == '/openai/v1/responses'
      end

      actual_sha = Digest::SHA256.hexdigest(sent_request.body)
      expect(sent_request.headers['X-Amz-Content-Sha256']).to eq(actual_sha)
    end
  end
end
