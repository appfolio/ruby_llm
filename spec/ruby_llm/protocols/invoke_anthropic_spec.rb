# frozen_string_literal: true

require 'spec_helper'
require 'aws-eventstream'

RSpec.describe RubyLLM::Protocols::InvokeAnthropic do
  let(:config) do
    RubyLLM::Configuration.new.tap do |c|
      c.bedrock_region = 'us-west-2'
      c.bedrock_api_key = 'static-key'
      c.bedrock_secret_key = 'static-secret'
    end
  end
  let(:provider) { RubyLLM::Providers::Bedrock.new(config) }
  let(:model_id) { 'arn:aws:bedrock:us-west-2:123456789012:application-inference-profile/abc123' }
  let(:model) { model_info(model_id) }
  let(:protocol) { described_class.new(provider, model) }

  def base
    'https://bedrock-runtime.us-west-2.amazonaws.com'
  end

  def escaped_path
    "/model/#{model_id.gsub('/', '%2F')}"
  end

  def model_info(id, max_output_tokens: 8192)
    RubyLLM::Model::Info.new(
      id: id, name: id, provider: 'bedrock', family: 'claude', created_at: nil,
      context_window: 200_000, max_output_tokens: max_output_tokens,
      modalities: { input: [], output: [] }, capabilities: [], pricing: {}, metadata: {}
    )
  end

  def user(text)
    RubyLLM::Message.new(role: :user, content: text)
  end

  def signed?(req)
    req.headers['Authorization']&.include?('us-west-2/bedrock/aws4_request')
  end

  def json_response(body)
    { status: 200, body: JSON.generate(body), headers: { 'Content-Type' => 'application/json' } }
  end

  def render(messages, **opts)
    protocol.render(messages, tools: opts.fetch(:tools, {}), temperature: nil, params: opts.fetch(:params, {}),
                              thinking: opts[:thinking], stream: opts.fetch(:stream, false))
  end

  describe 'URLs' do
    it 'targets /invoke and /invoke-with-response-stream with the model id as one escaped path segment' do
      expect(protocol.completion_url).to eq("#{escaped_path}/invoke")
      expect(protocol.send(:stream_url)).to eq("#{escaped_path}/invoke-with-response-stream")
    end
  end

  describe 'request body' do
    it 'carries anthropic_version and max_tokens, and never model or stream' do
      payload = render([user('hi')], stream: true)

      expect(payload[:anthropic_version]).to eq('bedrock-2023-05-31')
      expect(payload).not_to have_key(:model)
      expect(payload).not_to have_key(:stream)
      expect(payload[:max_tokens]).to eq(8192)
      expect(payload[:messages]).to eq([{ role: 'user', content: [{ type: 'text', text: 'hi' }] }])
    end

    it 'merges anthropic_beta from params into one de-duplicated list' do
      payload = render([user('hi')], params: { anthropic_beta: %w[a b a] })

      expect(payload[:anthropic_beta]).to eq(%w[a b])
    end

    it 'omits anthropic_beta when no betas are requested' do
      expect(render([user('hi')])).not_to have_key(:anthropic_beta)
    end

    it 'formats tools, tool_use and a folded tool_result + user turn' do
      tool = Class.new(RubyLLM::Tool) do
        def self.name = 'Weather'
        description 'Get weather'
        param :city, desc: 'City'
      end.new
      messages = [
        user('weather?'),
        RubyLLM::Message.new(role: :assistant, content: nil,
                             tool_calls: { 't1' => RubyLLM::ToolCall.new(id: 't1', name: 'weather',
                                                                         arguments: { 'city' => 'SF' }) }),
        RubyLLM::Message.new(role: :tool, content: 'sunny', tool_call_id: 't1'),
        user('thanks')
      ]

      payload = render(messages, tools: { weather: tool })

      expect(payload[:tools].first).to include(name: 'weather', description: 'Get weather')
      expect(payload[:messages][1]).to eq(
        role: 'assistant',
        content: [{ type: 'tool_use', id: 't1', name: 'weather', input: { 'city' => 'SF' } }]
      )
      expect(payload[:messages][2]).to eq(
        role: 'user',
        content: [
          { type: 'tool_result', tool_use_id: 't1', content: [{ type: 'text', text: 'sunny' }] },
          { type: 'text', text: 'thanks' }
        ]
      )
      expect(payload[:messages].size).to eq(3)
    end

    it 'sends images base64-inline' do
      content = RubyLLM::Content.new('look', [File.expand_path('../../fixtures/ruby.png', __dir__)])

      block = render([user(content)])[:messages].first[:content].last

      expect(block[:type]).to eq('image')
      expect(block[:source]).to include(type: 'base64', media_type: 'image/png')
      expect(block[:source][:data]).not_to be_empty
    end

    it 'replays Anthropic thinking blocks verbatim, before tool_use, even without thinking enabled' do
      blocks = [{ 'type' => 'thinking', 'thinking' => 'hmm', 'signature' => 'sig' },
                { 'type' => 'redacted_thinking', 'data' => 'opaque' }]
      assistant = RubyLLM::Message.new(
        role: :assistant, content: 'ok',
        thinking: RubyLLM::Thinking.build(text: 'hmm', signature: 'sig', blocks: blocks)
      )

      content = render([user('q'), assistant, user('next')])[:messages][1][:content]

      expect(content).to eq(blocks + [{ type: 'text', text: 'ok' }])
    end

    it 'translates Converse-captured reasoningContent blocks to the Messages shape' do
      blocks = [{ 'reasoningContent' => { 'reasoningText' => { 'text' => 'hmm', 'signature' => 'sig' } } },
                { 'reasoningContent' => { 'redactedContent' => 'opaque' } }]
      assistant = RubyLLM::Message.new(role: :assistant, content: 'ok',
                                       thinking: RubyLLM::Thinking.build(blocks: blocks))

      content = render([user('q'), assistant])[:messages][1][:content]

      expect(content.first(2)).to eq(
        [{ type: 'thinking', thinking: 'hmm', signature: 'sig' }, { type: 'redacted_thinking', data: 'opaque' }]
      )
    end

    it 'lifts nothing on its own: thinking/output_config arrive via params' do
      payload = render([user('q')], params: { thinking: { type: 'adaptive' }, output_config: { effort: 'high' } })

      expect(payload).to include(thinking: { type: 'adaptive' }, output_config: { effort: 'high' })
    end
  end

  describe 'sync completion' do
    it 'posts a SigV4-signed body and parses text, thinking blocks, tool calls and cache usage' do
      response = {
        'id' => 'msg_1', 'type' => 'message', 'role' => 'assistant', 'model' => 'claude-sonnet-5',
        'content' => [
          { 'type' => 'thinking', 'thinking' => 'let me look', 'signature' => 'sig-1' },
          { 'type' => 'redacted_thinking', 'data' => 'enc' },
          { 'type' => 'text', 'text' => 'Checking.' },
          { 'type' => 'tool_use', 'id' => 'toolu_1', 'name' => 'weather', 'input' => { 'city' => 'SF' } }
        ],
        'stop_reason' => 'tool_use',
        'usage' => { 'input_tokens' => 12, 'output_tokens' => 34, 'cache_read_input_tokens' => 1000,
                     'cache_creation_input_tokens' => 200 }
      }
      stub = stub_request(:post, "#{base}#{escaped_path}/invoke")
             .with do |req|
               body = JSON.parse(req.body)
               signed?(req) && body['anthropic_version'] == 'bedrock-2023-05-31' && !body.key?('model')
             end
             .to_return(json_response(response))

      message = protocol.complete([user('weather?')], tools: {}, temperature: nil)

      expect(stub).to have_been_requested
      expect(message.content).to eq('Checking.')
      expect(message.thinking.text).to eq('let me look')
      expect(message.thinking.signature).to eq('sig-1')
      expect(message.thinking.blocks).to eq(response['content'].first(2))
      expect(message.tool_calls['toolu_1'].arguments).to eq('city' => 'SF')
      expect(message.finish_reason).to eq('tool_use')
      expect(message.input_tokens).to eq(12)
      expect(message.output_tokens).to eq(34)
      expect(message.cached_tokens).to eq(1000)
      expect(message.cache_creation_tokens).to eq(200)
    end

    it 'maps a Bedrock throttling response to RateLimitError' do
      stub_request(:post, "#{base}#{escaped_path}/invoke")
        .to_return(status: 429, body: JSON.generate('message' => 'Too many requests'),
                   headers: { 'Content-Type' => 'application/json' })

      expect { protocol.complete([user('hi')], tools: {}, temperature: nil) }
        .to raise_error(RubyLLM::RateLimitError, /Too many requests/)
    end
  end

  describe 'streaming' do
    def event(payload)
      Aws::EventStream::Message.new(
        headers: {
          ':event-type' => Aws::EventStream::HeaderValue.new(value: 'chunk', type: 'string'),
          ':content-type' => Aws::EventStream::HeaderValue.new(value: 'application/json', type: 'string'),
          ':message-type' => Aws::EventStream::HeaderValue.new(value: 'event', type: 'string')
        },
        payload: StringIO.new(JSON.generate('bytes' => Base64.strict_encode64(JSON.generate(payload))))
      )
    end

    def exception(type, message)
      Aws::EventStream::Message.new(
        headers: {
          ':exception-type' => Aws::EventStream::HeaderValue.new(value: type, type: 'string'),
          ':content-type' => Aws::EventStream::HeaderValue.new(value: 'application/json', type: 'string'),
          ':message-type' => Aws::EventStream::HeaderValue.new(value: 'exception', type: 'string')
        },
        payload: StringIO.new(JSON.generate('message' => message))
      )
    end

    def wire(*messages)
      encoder = Aws::EventStream::Encoder.new
      messages.map { |message| encoder.encode(message) }.join
    end

    def stub_stream(body)
      stub_request(:post, "#{base}#{escaped_path}/invoke-with-response-stream")
        .with { |req| signed?(req) && !JSON.parse(req.body).key?('stream') }
        .to_return(status: 200, body: body, headers: { 'Content-Type' => 'application/vnd.amazon.eventstream' })
    end

    def stream(messages = [user('hi')])
      chunks = []
      message = protocol.complete(messages, tools: {}, temperature: nil) { |chunk| chunks << chunk }
      [message, chunks]
    end

    it 'decodes base64 chunk events into text, thinking blocks, tool calls and usage' do
      stub_stream(wire(
                    event('type' => 'message_start',
                          'message' => { 'model' => 'claude-sonnet-5',
                                         'usage' => { 'input_tokens' => 7, 'output_tokens' => 1,
                                                      'cache_read_input_tokens' => 900,
                                                      'cache_creation_input_tokens' => 100 } }),
                    event('type' => 'content_block_start', 'index' => 0,
                          'content_block' => { 'type' => 'thinking', 'thinking' => '' }),
                    event('type' => 'content_block_delta', 'index' => 0,
                          'delta' => { 'type' => 'thinking_delta', 'thinking' => 'plan ' }),
                    event('type' => 'content_block_delta', 'index' => 0,
                          'delta' => { 'type' => 'thinking_delta', 'thinking' => 'it' }),
                    event('type' => 'content_block_delta', 'index' => 0,
                          'delta' => { 'type' => 'signature_delta', 'signature' => 'sig-1' }),
                    event('type' => 'content_block_stop', 'index' => 0),
                    event('type' => 'content_block_start', 'index' => 1,
                          'content_block' => { 'type' => 'redacted_thinking', 'data' => 'enc' }),
                    event('type' => 'content_block_stop', 'index' => 1),
                    event('type' => 'content_block_start', 'index' => 2,
                          'content_block' => { 'type' => 'text', 'text' => '' }),
                    event('type' => 'content_block_delta', 'index' => 2,
                          'delta' => { 'type' => 'text_delta', 'text' => 'Hel' }),
                    event('type' => 'content_block_delta', 'index' => 2,
                          'delta' => { 'type' => 'text_delta', 'text' => 'lo' }),
                    event('type' => 'content_block_stop', 'index' => 2),
                    event('type' => 'content_block_start', 'index' => 3,
                          'content_block' => { 'type' => 'tool_use', 'id' => 'toolu_1', 'name' => 'weather',
                                               'input' => {} }),
                    event('type' => 'content_block_delta', 'index' => 3,
                          'delta' => { 'type' => 'input_json_delta', 'partial_json' => '{"city":' }),
                    event('type' => 'content_block_delta', 'index' => 3,
                          'delta' => { 'type' => 'input_json_delta', 'partial_json' => '"SF"}' }),
                    event('type' => 'content_block_stop', 'index' => 3),
                    event('type' => 'message_delta', 'delta' => { 'stop_reason' => 'tool_use' },
                          'usage' => { 'output_tokens' => 42 }),
                    event('type' => 'message_stop',
                          'amazon-bedrock-invocationMetrics' => { 'inputTokenCount' => 7 })
                  ))

      message, chunks = stream

      expect(chunks).not_to be_empty
      expect(message.content).to eq('Hello')
      expect(message.model_id).to eq('claude-sonnet-5')
      expect(message.thinking.text).to eq('plan it')
      expect(message.thinking.signature).to eq('sig-1')
      expect(message.thinking.blocks).to eq(
        [{ 'type' => 'thinking', 'thinking' => 'plan it', 'signature' => 'sig-1' },
         { 'type' => 'redacted_thinking', 'data' => 'enc' }]
      )
      expect(message.tool_calls['toolu_1'].name).to eq('weather')
      expect(message.tool_calls['toolu_1'].arguments).to eq('city' => 'SF')
      expect(message.finish_reason).to eq('tool_use')
      expect(message.input_tokens).to eq(7)
      expect(message.output_tokens).to eq(42)
      expect(message.cached_tokens).to eq(900)
      expect(message.cache_creation_tokens).to eq(100)
    end

    it 'keeps a signed thinking block left open at message_stop and drops an unsigned one' do
      stub_stream(wire(
                    event('type' => 'message_start', 'message' => { 'usage' => { 'input_tokens' => 1 } }),
                    event('type' => 'content_block_start', 'index' => 0,
                          'content_block' => { 'type' => 'thinking', 'thinking' => '' }),
                    event('type' => 'content_block_delta', 'index' => 0,
                          'delta' => { 'type' => 'signature_delta', 'signature' => 'sig-open' }),
                    event('type' => 'content_block_start', 'index' => 1,
                          'content_block' => { 'type' => 'thinking', 'thinking' => '' }),
                    event('type' => 'content_block_delta', 'index' => 1,
                          'delta' => { 'type' => 'thinking_delta', 'thinking' => 'half' }),
                    event('type' => 'message_stop')
                  ))

      message, = stream

      expect(message.thinking.blocks).to eq([{ 'type' => 'thinking', 'thinking' => '', 'signature' => 'sig-open' }])
    end

    it 'raises RateLimitError for a throttlingException event' do
      stub_stream(wire(
                    event('type' => 'message_start', 'message' => { 'usage' => { 'input_tokens' => 1 } }),
                    exception('throttlingException', 'Rate exceeded')
                  ))

      expect { stream }.to raise_error(RubyLLM::RateLimitError, /Rate exceeded/)
    end

    it 'raises BadRequestError for a validationException event' do
      stub_stream(wire(exception('validationException', 'bad thinking block')))

      expect { stream }.to raise_error(RubyLLM::BadRequestError, /bad thinking block/)
    end

    it 'raises OverloadedError for an Anthropic overloaded_error event' do
      stub_stream(wire(event('type' => 'error',
                             'error' => { 'type' => 'overloaded_error', 'message' => 'Overloaded' })))

      expect { stream }.to raise_error(RubyLLM::OverloadedError, /Overloaded/)
    end
  end

  describe 'Converse replay of InvokeModel-captured thinking' do
    it 'translates Anthropic thinking blocks to reasoningContent' do
      blocks = [{ 'type' => 'thinking', 'thinking' => 'hmm', 'signature' => 'sig' },
                { 'type' => 'redacted_thinking', 'data' => 'opaque' }]
      assistant = RubyLLM::Message.new(role: :assistant, content: 'ok',
                                       thinking: RubyLLM::Thinking.build(blocks: blocks))

      content = RubyLLM::Protocols::Converse::Chat.format_messages([assistant]).first[:content]

      expect(content.first(2)).to eq(
        [{ reasoningContent: { reasoningText: { text: 'hmm', signature: 'sig' } } },
         { reasoningContent: { redactedContent: 'opaque' } }]
      )
    end
  end
end
