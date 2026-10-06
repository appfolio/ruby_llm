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

    it 'uses /openai/v1/responses for openai.gpt-6.x ids' do
      %w[openai.gpt-6-luna openai.gpt-6-sol openai.gpt-6.1-sol].each do |id|
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

    it 'signs the compaction request at the frontier compact path' do
      body = { object: 'response.compaction', output: [{ type: 'compaction' }], usage: {} }.to_json
      stub = stub_request(:post, 'https://bedrock-mantle.us-west-2.api.aws/openai/v1/responses/compact')
             .with { |req| req.headers['Authorization']&.include?('us-west-2/bedrock-mantle/aws4_request') }
             .to_return(status: 200, body:, headers: { 'Content-Type' => 'application/json' })

      protocol = described_class.new(provider, model_info('openai.gpt-6-luna'))
      message = protocol.compact([RubyLLM::Message.new(role: :user, content: 'hi')])

      expect(stub).to have_been_requested
      expect(message.content.value['object']).to eq('response.compaction')
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

  describe 'streamed output items' do
    def stream(events)
      sse = events.map { |event| "data: #{event.to_json}\n\n" }.join
      stub_request(:post, 'https://bedrock-mantle.us-west-2.api.aws/openai/v1/responses')
        .to_return(status: 200, body: sse, headers: { 'Content-Type' => 'text/event-stream' })

      protocol = described_class.new(provider, model_info('openai.gpt-6-sol'))
      protocol.complete([RubyLLM::Message.new(role: :user, content: 'hi')], tools: {}, temperature: nil) { |_| nil }
    end

    let(:commentary) do
      { 'type' => 'message', 'phase' => 'commentary',
        'content' => [{ 'type' => 'output_text', 'text' => 'Checking.' }] }
    end
    let(:call) { { 'type' => 'function_call', 'call_id' => 'c1', 'name' => 'weather', 'arguments' => '{}' } }

    it 'sets OutputItems from the completed event output' do
      message = stream([
                         { type: 'response.output_text.delta', delta: 'Checking.' },
                         { type: 'response.output_item.done', output_index: 0, item: commentary },
                         { type: 'response.completed', response: { output: [commentary, call], usage: {} } }
                       ])

      expect(message.content).to be_a(RubyLLM::Protocols::Responses::OutputItems)
      expect(message.content.value).to eq([commentary, call])
      expect(message.content.commentary_text).to eq('Checking.')
    end

    it 'rebuilds OutputItems from output_item.done events when the completed event has no output' do
      message = stream([
                         { type: 'response.output_item.done', output_index: 1, item: call },
                         { type: 'response.output_item.done', output_index: 0, item: commentary },
                         { type: 'response.completed', response: { usage: {} } }
                       ])

      expect(message.content.value).to eq([commentary, call])
    end

    it 'leaves a plain streamed answer as a String' do
      answer = { 'type' => 'message', 'content' => [{ 'type' => 'output_text', 'text' => 'Hi' }] }
      message = stream([
                         { type: 'response.output_text.delta', delta: 'Hi' },
                         { type: 'response.output_item.done', output_index: 0, item: answer },
                         { type: 'response.completed', response: { output: [answer], usage: {} } }
                       ])

      expect(message.content).to eq('Hi')
    end
  end

  describe 'request fields passed through params' do
    let(:protocol) { described_class.new(provider, model_info('openai.gpt-6-sol')) }

    def render(params, schema: nil, thinking: nil)
      params = provider.send(:strip_converse_only_params, params)
      protocol.render([RubyLLM::Message.new(role: :user, content: 'hi')],
                      tools: {}, temperature: nil, params:, schema:, thinking:)
    end

    it 'sets none of them by default' do
      payload = render({})

      expect(payload.keys).not_to include(:context_management, :prompt_cache_key, :prompt_cache_retention,
                                          :truncation, :text, :reasoning, :tool_choice)
    end

    {
      context_management: [{ type: 'compaction', compact_threshold: 50_000 }],
      prompt_cache_key: 'chat-abc',
      prompt_cache_retention: '24h',
      truncation: 'auto',
      tool_choice: { type: 'allowed_tools', mode: 'auto', tools: [{ type: 'function', name: 'x' }] }
    }.each do |field, value|
      it "passes #{field} through to the body unchanged" do
        expect(render({ field => value })[field]).to eq(value)
      end
    end

    it 'merges text verbosity with the schema format' do
      schema = { name: 'response', schema: { type: 'object' }, strict: true }

      payload = render({ text: { verbosity: 'low' } }, schema:)

      expect(payload[:text]).to eq(verbosity: 'low',
                                   format: { type: 'json_schema', name: 'response', schema: { type: 'object' },
                                             strict: true })
    end

    it 'merges reasoning context with the thinking effort' do
      payload = render({ reasoning: { context: 'all_turns' } }, thinking: RubyLLM::Thinking::Config.new(effort: 'high'))

      expect(payload[:reasoning]).to eq(effort: 'high', context: 'all_turns')
    end
  end
end
