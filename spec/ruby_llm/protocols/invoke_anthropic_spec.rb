# frozen_string_literal: true

require 'spec_helper'
require 'aws-eventstream'

class InvokeAnthropicWeather < RubyLLM::Tool
  description 'Get the current weather for a city'
  parameter :city, description: 'City name'

  def execute(city:)
    "Sunny in #{city}"
  end
end

class InvokeAnthropicTime < RubyLLM::Tool
  description 'Get the current time'

  def execute
    'noon'
  end
end

RSpec.describe RubyLLM::Protocols::InvokeAnthropic do
  include_context 'with configured RubyLLM'

  let(:provider) { RubyLLM::Providers::Bedrock.new(RubyLLM.config) }
  let(:model) { RubyLLM.models.find(model_id, provider: :bedrock) }
  let(:protocol) { described_class.new(provider, model) }

  def model_id = 'us.anthropic.claude-opus-5-5'

  def runtime = "https://bedrock-runtime.us-west-2.amazonaws.com/model/#{model_id}"

  def tools = { weather: InvokeAnthropicWeather.new, time: InvokeAnthropicTime.new }

  def weather = InvokeAnthropicWeather.new.name

  def time = InvokeAnthropicTime.new.name

  def user_message = RubyLLM::Message.new(role: :user, content: 'Hi')

  def chat
    RubyLLM.chat(model: model_id, provider: :bedrock, protocol: :invoke_anthropic)
  end

  def render(messages = [user_message], **options)
    protocol.render(messages, tools: {}, temperature: nil, **options)
  end

  def reply_body(**fields)
    {
      'id' => 'msg_1', 'type' => 'message', 'role' => 'assistant', 'model' => 'claude-opus-5-5',
      'content' => [{ 'type' => 'text', 'text' => 'OK' }], 'stop_reason' => 'end_turn',
      'usage' => { 'input_tokens' => 10, 'output_tokens' => 2 }
    }.merge(fields)
  end

  def stub_invoke(body = reply_body, &)
    stub = stub_request(:post, "#{runtime}/invoke")
    stub = stub.with(&) if block_given?
    stub.to_return(status: 200, body: JSON.generate(body), headers: { 'Content-Type' => 'application/json' })
  end

  def frame(headers, payload)
    message = Aws::EventStream::Message.new(
      headers: headers.transform_values { |value| Aws::EventStream::HeaderValue.new(value:, type: 'string') },
      payload: StringIO.new(payload)
    )
    Aws::EventStream::Encoder.new.encode_message(message)
  end

  def chunk_frame(event)
    frame({ ':message-type' => 'event', ':event-type' => 'chunk', ':content-type' => 'application/json' },
          JSON.generate('bytes' => Base64.strict_encode64(JSON.generate(event))))
  end

  def eventstream(events)
    events.map { |event| event.is_a?(String) ? event : chunk_frame(event) }.join
  end

  def stub_invoke_stream(events, &)
    stub = stub_request(:post, "#{runtime}/invoke-with-response-stream")
    stub = stub.with(&) if block_given?
    stub.to_return(status: 200, body: eventstream(events),
                   headers: { 'Content-Type' => 'application/vnd.amazon.eventstream' })
  end

  def message_start(usage: { 'input_tokens' => 50, 'output_tokens' => 1 }, **fields)
    { 'type' => 'message_start',
      'message' => { 'id' => 'msg_1', 'type' => 'message', 'role' => 'assistant', 'model' => 'claude-opus-5-5',
                     'content' => [], 'stop_reason' => nil, 'usage' => usage }.merge(fields) }
  end

  def message_end(stop_reason: 'end_turn', usage: { 'output_tokens' => 5 }, **fields)
    [{ 'type' => 'message_delta', 'delta' => { 'stop_reason' => stop_reason }, 'usage' => usage }.merge(fields),
     { 'type' => 'message_stop' }]
  end

  def text_block(index, text)
    [{ 'type' => 'content_block_start', 'index' => index, 'content_block' => { 'type' => 'text', 'text' => '' } },
     { 'type' => 'content_block_delta', 'index' => index, 'delta' => { 'type' => 'text_delta', 'text' => text } },
     { 'type' => 'content_block_stop', 'index' => index }]
  end

  def tool_use_block(index, type:, id:, name:, json:)
    [{ 'type' => 'content_block_start', 'index' => index,
       'content_block' => { 'type' => type, 'id' => id, 'name' => name, 'input' => {} } },
     { 'type' => 'content_block_delta', 'index' => index,
       'delta' => { 'type' => 'input_json_delta', 'partial_json' => json } },
     { 'type' => 'content_block_stop', 'index' => index }]
  end

  def stream(messages = [user_message], **options)
    chunks = []
    message = protocol.complete(messages, tools: {}, temperature: nil, **options) { |chunk| chunks << chunk }
    [message, chunks]
  end

  describe 'protocol resolution' do
    it 'resolves protocol: :invoke_anthropic to this class' do
      expect(provider.send(:resolve_protocol, :invoke_anthropic, model)).to be <= described_class
      expect(chat.ask_later('Hi').render).to include(anthropic_version: 'bedrock-2023-05-31')
      expect(RubyLLM.chat(model: model_id, provider: :bedrock).with_model(model_id, protocol: :invoke_anthropic)
                    .ask_later('Hi').render).to include(:anthropic_version)
    end

    it 'keeps Claude on Converse when no protocol is given' do
      expect(provider.send(:resolve_protocol, nil, model)).to eq(provider.protocols[:converse])
      expect(provider.protocol_for(model)).to be <= RubyLLM::Protocols::Converse
      expect(RubyLLM.chat(model: model_id, provider: :bedrock).ask_later('Hi').render).to include(:inferenceConfig)
    end
  end

  describe 'the rendered body' do
    it 'carries anthropic_version and neither model nor stream' do
      body = render(stream: true)

      expect(body.keys.first).to eq(:anthropic_version)
      expect(body[:anthropic_version]).to eq('bedrock-2023-05-31')
      expect(body).not_to include(:model, :stream)
    end

    it 'omits anthropic_beta when no beta is needed' do
      expect(render).not_to have_key(:anthropic_beta)
    end

    it 'joins caller betas and the compaction beta into one unique list' do
      body = render(compaction: { at: 50_000 },
                    provider_options: { anthropic_beta: %w[context-1m compact-2026-01-12 context-1m] })

      expect(body[:anthropic_beta]).to eq(%w[context-1m compact-2026-01-12])
      expect(body[:context_management][:edits].first).to include(type: 'compact_20260112')
    end

    it 'adds the compaction beta when compaction is the only reason' do
      expect(render(compaction: {})[:anthropic_beta]).to eq(['compact-2026-01-12'])
    end

    it 'lifts additionalModelRequestFields to the top level' do
      fields = { 'thinking' => { 'type' => 'adaptive' }, 'output_config' => { 'effort' => 'medium' } }

      [{ additionalModelRequestFields: fields }, { 'additionalModelRequestFields' => fields }].each do |options|
        body = render(provider_options: options)

        expect(body).to include(thinking: { type: 'adaptive' }, output_config: { effort: 'medium' })
        expect(body).not_to include(:additionalModelRequestFields, 'additionalModelRequestFields')
      end
    end

    it 'marks exactly the deferred tools and puts server tools first' do
      search = { type: 'tool_search_tool_regex_20251119', name: 'tool_search_tool_regex' }
      body = protocol.render([user_message], tools:, temperature: nil,
                                             provider_options: { deferred_tools: [weather], server_tools: [search] })

      expect(body[:tools].map { |tool| tool[:name] }).to eq(['tool_search_tool_regex', weather, time])
      expect(body[:tools].first).to eq(search)
      expect(body[:tools][1][:defer_loading]).to be(true)
      expect(body[:tools][2]).not_to have_key(:defer_loading)
      expect(body).not_to include(:deferred_tools, :server_tools)
    end

    it 'raises for a deferred tool name that names no chat tool' do
      expect do
        protocol.render([user_message], tools:, temperature: nil,
                                        provider_options: { deferred_tools: [weather, 'nope'] })
      end.to raise_error(ArgumentError, /deferred_tools names no chat tool: nope/)
    end

    it 'passes context_management and withheld_tools through' do
      edits = { edits: [{ type: 'clear_tool_uses_20250919', keep: { type: 'tool_uses', value: 3 } }] }
      body = render(provider_options: { context_management: edits, withheld_tools: ['weather'] })

      expect(body[:context_management]).to eq(edits)
      expect(body[:withheld_tools]).to eq(['weather'])
    end

    it 'renders adaptive thinking with the effort for this model' do
      body = render(thinking: RubyLLM::Thinking::Config.new(effort: :high))

      expect(body[:thinking]).to eq(type: 'adaptive')
      expect(body[:output_config]).to eq(effort: 'high')
    end

    it 'keeps leading system messages in system and a later one in place' do
      messages = [
        RubyLLM::Message.new(role: :system, content: 'Be brief.'),
        user_message,
        RubyLLM::Message.new(role: :system, content: 'Now be formal.'),
        RubyLLM::Message.new(role: :user, content: 'Hello')
      ]
      body = render(messages)

      expect(body[:system]).to eq([{ type: 'text', text: 'Be brief.' }])
      expect(body[:messages].map { |message| message[:role] }).to eq(%w[user system user])
      expect(body[:messages][1][:content]).to eq([{ text: 'Now be formal.', type: 'text' }])
    end

    it 'sorts content keys at every depth' do
      raw = [{ 'type' => 'server_tool_use', 'name' => 'tool_search_tool_regex', 'id' => 'srv_1',
               'input' => { 'pattern' => 'w', 'limit' => 2 } }]
      message = RubyLLM::Message.new(role: :assistant, content: '', raw_content: raw)
      block = render([user_message, message])[:messages].last[:content].first

      expect(block.keys).to eq(%w[id input name type])
      expect(block['input'].keys).to eq(%w[limit pattern])
    end

    it 'renders the same JSON from symbol keys and from string keys in another order' do
      symbols = [{ type: 'server_tool_use', id: 'srv_1', input: { query: 'w', limit: 2 }, name: 'tool_search' }]
      strings = [{ 'name' => 'tool_search', 'input' => { 'limit' => 2, 'query' => 'w' }, 'id' => 'srv_1',
                   'type' => 'server_tool_use' }]
      json = [symbols, strings].map do |raw|
        message = RubyLLM::Message.new(role: :assistant, content: '', raw_content: raw)
        JSON.generate(render([user_message, message])[:messages])
      end

      expect(json.first).to eq(json.last)
    end

    it 'renders a cache_until_here breakpoint on the message' do
      message = RubyLLM::Message.new(role: :user, content: 'Long context').cache_until_here
      block = render([message])[:messages].first[:content].last

      expect(block[:cache_control]).to eq(type: 'ephemeral')
    end

    it 'rejects provider tools Bedrock cannot run' do
      expect { render(provider_tools: [:web_search]) }.to raise_error(RubyLLM::UnsupportedServerToolError)
    end
  end

  describe 'before_request hooks' do
    it 'see the finished body' do
      seen = nil
      hook = lambda do |body|
        seen = { version: body[:anthropic_version], betas: body[:anthropic_beta],
                 deferred: body[:tools].find { |tool| tool[:name] == weather }[:defer_loading],
                 withheld: body[:withheld_tools], keys: body.keys }
      end
      protocol.render([user_message], tools:, temperature: nil, compaction: {}, before_request: [hook],
                                      provider_options: { deferred_tools: [weather], withheld_tools: [time] })

      expect(seen).to include(version: 'bedrock-2023-05-31', betas: ['compact-2026-01-12'], deferred: true,
                              withheld: [time])
      expect(seen[:keys]).not_to include(:model, :stream, :deferred_tools)
    end

    it 'edit the body that is signed and posted' do
      hook = lambda do |body|
        body.delete(:withheld_tools)
        body[:metadata] = { user_id: 'hooked' }
      end
      request = stub_invoke do |req|
        body = JSON.parse(req.body)
        body['metadata'] == { 'user_id' => 'hooked' } && !body.key?('withheld_tools') &&
          req.headers['X-Amz-Content-Sha256'] == Digest::SHA256.hexdigest(req.body)
      end

      protocol.complete([user_message], tools: {}, temperature: nil, before_request: [hook],
                                        provider_options: { withheld_tools: [time] })

      expect(request).to have_been_requested
    end
  end

  describe 'requests' do
    it 'posts sync replies to /invoke without an anthropic-beta header' do
      [nil, {}].each do |compaction|
        WebMock.reset!
        request = stub_invoke { |req| req.headers.keys.none? { |name| name.casecmp?('anthropic-beta') } }

        protocol.complete([user_message], tools: {}, temperature: nil, compaction:,
                                          headers: { 'anthropic-beta' => 'files-api-2025-04-14' })

        expect(request).to have_been_requested
      end
    end

    it 'posts streams to /invoke-with-response-stream without an anthropic-beta header' do
      [nil, {}].each do |compaction|
        WebMock.reset!
        request = stub_invoke_stream([message_start, *text_block(0, 'OK'), *message_end]) do |req|
          req.headers['Accept'] == 'application/vnd.amazon.eventstream' &&
            req.headers.keys.none? { |name| name.casecmp?('anthropic-beta') }
        end

        stream(compaction:, headers: { 'Anthropic-Beta' => 'files-api-2025-04-14' })

        expect(request).to have_been_requested
      end
    end

    it 'escapes an inference profile ARN into one path segment' do
      arn = 'arn:aws:bedrock:us-west-2:123456789012:application-inference-profile/abc123'
      arn_protocol = described_class.new(provider, RubyLLM::Model.default(arn, 'bedrock'))

      expect(arn_protocol.completion_url).to eq("/model/#{arn.sub('/', '%2F')}/invoke")
      expect(arn_protocol.send(:stream_url)).to eq("/model/#{arn.sub('/', '%2F')}/invoke-with-response-stream")
    end

    it 'rejects token counting and manual compaction' do
      expect { protocol.count_tokens([user_message], tools: {}) }.to raise_error(RubyLLM::Error, /InvokeModel/)
      expect { protocol.compact([user_message]) }.to raise_error(RubyLLM::Error, /InvokeModel/)
    end
  end

  describe 'streaming' do
    it 'streams a text reply' do
      stub_invoke_stream([message_start, *text_block(0, 'Hello'), *message_end(usage: { 'input_tokens' => 42,
                                                                                        'output_tokens' => 5 })])

      message, chunks = stream

      expect(chunks.filter_map(&:content).join).to eq('Hello')
      expect(message.content).to eq('Hello')
      expect(message.finish_reason).to eq(:stop)
      expect(message.tokens.input).to eq(42)
      expect(message.tokens.output).to eq(5)
    end

    it 'streams a tool call' do
      stub_invoke_stream([message_start, *text_block(0, 'Checking.'),
                          *tool_use_block(1, type: 'tool_use', id: 'toolu_1', name: 'weather',
                                             json: '{"city":"Paris"}'),
                          *message_end(stop_reason: 'tool_use')])

      message, = stream

      expect(message.tool_calls.values.map { |call| [call.id, call.name, call.arguments] })
        .to eq([['toolu_1', 'weather', { 'city' => 'Paris' }]])
      expect(message.finish_reason).to eq(:tool_calls)
    end

    it 'streams a compaction block' do
      stub_invoke_stream([
                           message_start,
                           { 'type' => 'content_block_start', 'index' => 0,
                             'content_block' => { 'type' => 'compaction', 'content' => nil } },
                           { 'type' => 'content_block_delta', 'index' => 0,
                             'delta' => { 'type' => 'compaction_delta', 'content' => 'Summary so far.' } },
                           { 'type' => 'content_block_stop', 'index' => 0 },
                           *text_block(1, 'Done.'),
                           *message_end
                         ])

      message, = stream(compaction: {})

      expect(message.raw_content.first).to eq('type' => 'compaction', 'content' => 'Summary so far.')
      expect(message.server_tool_calls.map(&:type)).to eq(['compaction'])
      expect(message.content).to eq('Done.')
    end

    it 'streams a server_tool_use with its tool_search_tool_result' do
      result = { 'type' => 'tool_search_tool_result', 'tool_use_id' => 'srv_1',
                 'content' => { 'type' => 'tool_search_tool_search_result',
                                'tool_references' => [{ 'type' => 'tool_reference', 'tool_name' => 'weather' }] } }
      stub_invoke_stream([
                           message_start,
                           *tool_use_block(0, type: 'server_tool_use', id: 'srv_1', name: 'tool_search_tool_regex',
                                              json: '{"pattern":"weather"}'),
                           { 'type' => 'content_block_start', 'index' => 1, 'content_block' => result },
                           { 'type' => 'content_block_stop', 'index' => 1 },
                           *tool_use_block(2, type: 'tool_use', id: 'toolu_2', name: 'weather',
                                              json: '{"city":"Oslo"}'),
                           *message_end(stop_reason: 'tool_use')
                         ])

      message, = stream

      expect(message.raw_content.map { |block| block['type'] })
        .to eq(%w[server_tool_use tool_search_tool_result tool_use])
      expect(message.raw_content.first['input']).to eq('pattern' => 'weather')
      expect(message.server_tool_calls.map(&:type)).to eq(%w[server_tool_use tool_search_tool_result])
      expect(message.tool_calls.values.map(&:name)).to eq(['weather'])
    end

    it 'raises the rate limit error for a throttlingException frame' do
      throttled = frame({ ':message-type' => 'exception', ':exception-type' => 'throttlingException' },
                        JSON.generate('message' => 'Too many requests'))
      stub_invoke_stream([message_start, throttled])

      expect { stream }.to raise_error(RubyLLM::RateLimitError, /Too many requests/)
    end

    it 'raises the rate limit error for a token throttle frame' do
      throttled = frame({ ':message-type' => 'exception', ':exception-type' => 'throttlingException' },
                        JSON.generate('message' => 'Too many tokens, please wait before trying again.'))
      stub_invoke_stream([message_start, throttled])

      expect { stream }.to raise_error(RubyLLM::RateLimitError, /Too many tokens/)
    end

    it 'raises the overloaded error for an overloaded_error event' do
      stub_invoke_stream([message_start,
                          { 'type' => 'error',
                            'error' => { 'type' => 'overloaded_error', 'message' => 'Overloaded' } }])

      expect { stream }.to raise_error(RubyLLM::OverloadedError, /Overloaded/)
    end

    it 'raises a server error for any other Anthropic error event' do
      stub_invoke_stream([{ 'type' => 'error', 'error' => { 'type' => 'api_error', 'message' => 'Broken' } }])

      expect { stream }.to raise_error(RubyLLM::ServerError, /Broken/)
    end

    it 'marks the stream started once the first chunk reaches the caller' do
      progress = {}
      accumulator = RubyLLM::Protocol::StreamAccumulator.new
      bytes = eventstream([message_start])
      yielded = []

      protocol.send(:parse_stream_chunk, Aws::EventStream::Decoder.new, bytes, accumulator, progress) do |chunk|
        yielded << progress.dup
        chunk
      end

      expect(progress[:started]).to be(true)
      expect(yielded).to eq([{ started: true }])
    end
  end

  describe 'provider_data' do
    def context_management
      { 'applied_edits' => [{ 'type' => 'clear_tool_uses_20250919', 'cleared_tool_uses' => 2 }] }
    end

    def iterations
      [{ 'type' => 'compaction', 'input_tokens' => 900, 'output_tokens' => 80 },
       { 'type' => 'message', 'input_tokens' => 120, 'output_tokens' => 9 }]
    end

    def cache_creation = { 'ephemeral_5m_input_tokens' => 0, 'ephemeral_1h_input_tokens' => 300 }
    def input_transformations = [{ 'type' => 'thinking_dropped', 'count' => 1 }]

    def expected
      { 'context_management' => context_management, 'iterations' => iterations,
        'cache_creation' => cache_creation, 'input_transformations' => input_transformations }
    end

    it 'carries the four reply fields from a sync body' do
      stub_invoke(reply_body('context_management' => context_management,
                             'input_transformations' => input_transformations,
                             'usage' => { 'input_tokens' => 120, 'output_tokens' => 9,
                                          'iterations' => iterations, 'cache_creation' => cache_creation }))

      message = protocol.complete([user_message], tools: {}, temperature: nil)

      expect(message.provider_data).to eq(expected)
    end

    it 'carries the four reply fields from a stream' do
      stub_invoke_stream([
                           message_start(input_transformations:),
                           *text_block(0, 'OK'),
                           *message_end(context_management:,
                                        usage: { 'input_tokens' => 120, 'output_tokens' => 9,
                                                 'iterations' => iterations, 'cache_creation' => cache_creation })
                         ])

      message, = stream

      expect(message.provider_data).to eq(expected)
      expect(message.tokens.input).to eq(1020)
    end

    it 'is empty on a reply without those fields' do
      stub_invoke

      expect(protocol.complete([user_message], tools: {}, temperature: nil).provider_data).to eq({})
    end

    it 'is nil on a Converse reply' do
      stub_request(:post, "#{runtime}/converse").to_return(
        status: 200, headers: { 'Content-Type' => 'application/json' },
        body: JSON.generate('output' => { 'message' => { 'role' => 'assistant', 'content' => [{ 'text' => 'OK' }] } },
                            'stopReason' => 'end_turn', 'usage' => { 'inputTokens' => 3, 'outputTokens' => 1 })
      )

      message = RubyLLM::Protocols::Converse.new(provider, model).complete([user_message], tools: {}, temperature: nil)

      expect(message.content).to eq('OK')
      expect(message.provider_data).to be_nil
    end
  end

  describe 'provider_data across pause_turn segments' do
    def edit(cleared) = { 'type' => 'clear_tool_uses_20250919', 'cleared_tool_uses' => cleared }

    def iteration(type, input) = { 'type' => type, 'input_tokens' => input, 'output_tokens' => 1 }

    def json_reply(body) = { status: 200, body: JSON.generate(body), headers: { 'Content-Type' => 'application/json' } }

    def stream_reply(events)
      { status: 200, body: eventstream(events), headers: { 'Content-Type' => 'application/vnd.amazon.eventstream' } }
    end

    it 'merges the fields of every segment of a sync turn' do
      first = reply_body('stop_reason' => 'pause_turn',
                         'context_management' => { 'applied_edits' => [edit(1)], 'state' => 'first' },
                         'input_transformations' => [{ 'type' => 'thinking_dropped' }],
                         'usage' => { 'input_tokens' => 10, 'output_tokens' => 1,
                                      'iterations' => [iteration('compaction', 900)],
                                      'cache_creation' => { 'ephemeral_5m_input_tokens' => 10,
                                                            'ephemeral_1h_input_tokens' => 100 } })
      second = reply_body('context_management' => { 'applied_edits' => [edit(2)], 'state' => 'last' },
                          'usage' => { 'input_tokens' => 20, 'output_tokens' => 2,
                                       'iterations' => [iteration('message', 120), iteration('message', 130)],
                                       'cache_creation' => { 'ephemeral_5m_input_tokens' => 1,
                                                             'ephemeral_1h_input_tokens' => 200 } })
      stub_request(:post, "#{runtime}/invoke").to_return(json_reply(first)).then.to_return(json_reply(second))

      message = protocol.complete([user_message], tools: {}, temperature: nil)

      expect(message.provider_data).to eq(
        'context_management' => { 'applied_edits' => [edit(1), edit(2)], 'state' => 'last' },
        'iterations' => [iteration('compaction', 900), iteration('message', 120), iteration('message', 130)],
        'cache_creation' => { 'ephemeral_5m_input_tokens' => 11, 'ephemeral_1h_input_tokens' => 300 },
        'input_transformations' => [{ 'type' => 'thinking_dropped' }]
      )
    end

    it 'merges the fields of every segment of a streamed turn' do
      first = [message_start(input_transformations: [{ 'type' => 'thinking_dropped' }]), *text_block(0, 'Part one. '),
               *message_end(stop_reason: 'pause_turn',
                            usage: { 'output_tokens' => 3, 'iterations' => [iteration('message', 400)],
                                     'cache_creation' => { 'ephemeral_5m_input_tokens' => 5 } })]
      second = [message_start(input_transformations: [{ 'type' => 'thinking_flagged' }]), *text_block(0, 'Part two.'),
                *message_end(usage: { 'output_tokens' => 4, 'iterations' => [iteration('message', 450)],
                                      'cache_creation' => { 'ephemeral_5m_input_tokens' => 7,
                                                            'ephemeral_1h_input_tokens' => 9 } })]
      stub_request(:post, "#{runtime}/invoke-with-response-stream")
        .to_return(stream_reply(first)).then.to_return(stream_reply(second))

      message, = stream

      expect(message.content).to eq('Part one. Part two.')
      expect(message.provider_data).to eq(
        'iterations' => [iteration('message', 400), iteration('message', 450)],
        'cache_creation' => { 'ephemeral_5m_input_tokens' => 12, 'ephemeral_1h_input_tokens' => 9 },
        'input_transformations' => [{ 'type' => 'thinking_dropped' }, { 'type' => 'thinking_flagged' }]
      )
    end
  end
end
