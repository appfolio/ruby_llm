# frozen_string_literal: true

require 'spec_helper'

RSpec.describe RubyLLM::Streaming do
  let(:test_obj) do
    Object.new.tap do |obj|
      obj.extend(described_class)
      obj.define_singleton_method(:build_chunk) { |data| "chunk:#{data['x']}" }
      obj.define_singleton_method(:parse_error) do |response|
        response.body.dig('error', 'message') if response.body.is_a?(Hash)
      end
    end
  end

  let(:env) { Faraday::Env.from(status: 200) }
  let(:parsed_error) { { 'error' => { 'message' => 'Rate limit exceeded' } } }

  before do
    stub_const('Faraday::VERSION', '2.0.0')
  end

  it 'skips non-hash SSE payloads' do
    yielded_chunks = []
    handler = test_obj.send(:handle_stream) { |chunk| yielded_chunks << chunk }

    expect { handler.call("data: true\n\n", 0, env) }.not_to raise_error
    expect(yielded_chunks).to eq([])
  end

  it 'processes hash SSE payloads' do
    yielded_chunks = []
    handler = test_obj.send(:handle_stream) { |chunk| yielded_chunks << chunk }

    handler.call("data: {\"x\":\"ok\"}\n\n", 0, env)

    expect(yielded_chunks).to eq(['chunk:ok'])
  end

  it 'prefers the failed HTTP response status over a generic parsed stream status' do
    failed_env = Faraday::Env.from(status: 429)

    response = test_obj.send(:build_stream_error_response, parsed_error, failed_env, 500)

    expect(response.status).to eq(429)
    expect(response.body).to eq(parsed_error)
  end

  it 'uses the parsed stream status when the HTTP response status is successful' do
    response = test_obj.send(:build_stream_error_response, parsed_error, env, 529)

    expect(response.status).to eq(529)
  end

  it 'stores parsed streaming errors on the response env before raising' do
    failed_env = Faraday::Env.from(status: 429)

    expect do
      test_obj.send(:raise_stream_error, parsed_error.to_json, parsed_error, failed_env)
    end.to raise_error(RubyLLM::RateLimitError, 'Rate limit exceeded')

    response = failed_env[:streaming_error_response]
    expect(response.status).to eq(429)
    expect(response.body).to eq(parsed_error)
  end

  describe 'skipped SSE events' do
    let(:skipped) { [] }

    before do
      allow(RubyLLM.logger).to receive(:warn)
      RubyLLM.config.on_stream_parse_error = ->(event) { skipped << event }
    end

    after { RubyLLM.config.on_stream_parse_error = nil }

    def feed(sse)
      chunks = []
      handler = test_obj.send(:handle_stream) { |chunk| chunks << chunk }
      handler.call(sse, 0, env)
      chunks
    end

    it 'reports data that is not valid JSON with its SSE event type, and keeps streaming' do
      chunks = feed("event: content_block_delta\ndata: {\"x\":\"secret\n\ndata: {\"x\":\"ok\"}\n\n")

      expect(chunks).to eq(['chunk:ok'])
      expect(skipped.map(&:to_h)).to eq(
        [{ provider: nil, model: nil, site: :sse_data, event_type: 'content_block_delta', byte_size: 12,
           error_class: 'JSON::ParserError' }]
      )
      expect(RubyLLM.logger).to have_received(:warn)
        .with(a_string_including('site=sse_data event_type=content_block_delta byte_size=12'))
      expect(RubyLLM.logger).not_to have_received(:warn).with(a_string_including('secret'))
    end

    it 'reports data that decodes to something other than a JSON object' do
      expect(feed("data: true\n\n")).to eq([])

      expect(skipped.map(&:to_h)).to eq(
        [{ provider: nil, model: nil, site: :sse_data, event_type: nil, byte_size: 4, error_class: nil }]
      )
    end

    it 'does not report the [DONE] sentinel' do
      feed("data: [DONE]\n\n")

      expect(skipped).to be_empty
      expect(RubyLLM.logger).not_to have_received(:warn)
    end

    it 'still logs a skip when no hook is configured' do
      RubyLLM.config.on_stream_parse_error = nil

      expect { feed("data: {\n\n") }.not_to raise_error
      expect(RubyLLM.logger).to have_received(:warn).with(a_string_including('site=sse_data'))
    end
  end

  describe 'skipped event count on a streamed message' do
    let(:config) { RubyLLM::Configuration.new.tap { |c| c.openai_api_key = 'test' } }
    let(:protocol) do
      model = RubyLLM::Model::Info.new(id: 'gpt-test', name: 'gpt-test', provider: 'openai', created_at: nil,
                                       context_window: 128_000, max_output_tokens: 4096,
                                       modalities: { input: [], output: [] }, capabilities: [], pricing: {},
                                       metadata: {})
      RubyLLM::Protocols::ChatCompletions.new(RubyLLM::Providers::OpenAI.new(config), model)
    end

    before { allow(RubyLLM.logger).to receive(:warn) }

    def record_skips
      [].tap { |skipped| config.on_stream_parse_error = ->(event) { skipped << event } }
    end

    def stream(*data_lines)
      body = data_lines.map { |line| "data: #{line}\n\n" }.join
      stub_request(:post, 'https://api.openai.com/v1/chat/completions')
        .to_return(status: 200, body: body, headers: { 'Content-Type' => 'text/event-stream' })
      protocol.complete([RubyLLM::Message.new(role: :user, content: 'hi')], tools: {}, temperature: nil) { |_| nil }
    end

    it 'records how many events were skipped and reports provider and model' do
      skipped = record_skips
      message = stream('{"model":"gpt-test","choices":[{"delta":{"content":"Hel"}}]}',
                       '{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1"',
                       '{"choices":[{"delta":{"content":"lo"}}]}',
                       '[DONE]')

      expect(message.content).to eq('Hello')
      expect(message.skipped_stream_events).to eq(1)
      expect(skipped.map { |event| [event.provider, event.model, event.site] })
        .to eq([['openai', 'gpt-test', :sse_data]])
    end

    it 'records zero when every event decodes' do
      skipped = record_skips
      message = stream('{"model":"gpt-test","choices":[{"delta":{"content":"Hi"}}]}', '[DONE]')

      expect(message.skipped_stream_events).to eq(0)
      expect(skipped).to be_empty
    end
  end
end
