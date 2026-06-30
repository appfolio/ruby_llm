# frozen_string_literal: true

require 'spec_helper'

RSpec.describe RubyLLM::Protocols::BedrockInvokeModel do
  # ────────────────────────────────────────────────────────────────────────────
  # Helpers
  # ────────────────────────────────────────────────────────────────────────────

  let(:model) do
    instance_double(
      RubyLLM::Model::Info,
      id: 'us.anthropic.claude-sonnet-4-5-20250929-v1:0',
      max_tokens: 8096,
      metadata: {}
    )
  end
  let(:protocol) { build_protocol }
  let(:base_args) do
    {
      tools: {},
      temperature: nil,
      model: model,
      stream: false
    }
  end

  def build_config(overrides = {})
    RubyLLM::Configuration.new.tap do |cfg|
      cfg.bedrock_region = 'us-east-1'
      cfg.bedrock_api_key = 'key'
      cfg.bedrock_secret_key = 'secret'
      overrides.each { |k, v| cfg.public_send(:"#{k}=", v) }
    end
  end

  def build_protocol(config_overrides = {})
    config = build_config(config_overrides)
    protocol = described_class.allocate
    protocol.instance_variable_set(:@config, config)
    protocol.instance_variable_set(:@model, model)
    protocol
  end

  def render_payload(messages = [], **overrides)
    protocol.send(:render_payload, messages, **base_args, **overrides)
  end

  # ────────────────────────────────────────────────────────────────────────────
  # render_payload
  # ────────────────────────────────────────────────────────────────────────────

  describe '#render_payload' do
    it 'includes anthropic_version' do
      payload = render_payload
      expect(payload[:anthropic_version]).to eq('bedrock-2023-05-31')
    end

    it 'puts max_tokens at the top level (not inside inferenceConfig)' do
      payload = render_payload
      expect(payload[:max_tokens]).to eq(8096)
      expect(payload).not_to have_key(:inferenceConfig)
    end

    it 'does not include a model key in the body' do
      payload = render_payload
      expect(payload).not_to have_key(:model)
    end

    it 'falls back to 4096 when model.max_tokens is nil' do
      allow(model).to receive(:max_tokens).and_return(nil)
      payload = render_payload
      expect(payload[:max_tokens]).to eq(4096)
    end

    it 'includes messages in the payload' do
      msg = RubyLLM::Message.new(role: :user, content: 'Hello')
      payload = render_payload([msg])
      expect(payload[:messages]).to be_an(Array)
      expect(payload[:messages].first[:role]).to eq('user')
    end

    context 'with system messages' do
      it 'promotes system messages to the top-level system field' do
        system_msg = RubyLLM::Message.new(role: :system, content: 'You are helpful.')
        user_msg = RubyLLM::Message.new(role: :user, content: 'Hi')
        payload = render_payload([system_msg, user_msg])

        expect(payload).to have_key(:system)
        expect(payload[:messages].none? { |m| m[:role] == 'system' }).to be true
      end
    end

    context 'with tools' do
      let(:tool) do
        t = instance_double(RubyLLM::Tool)
        allow(t).to receive_messages(name: 'calculate', description: 'Does math', params_schema: nil, parameters: [],
                                     provider_params: {})
        t
      end

      it 'emits tools in Anthropic input_schema shape (not Converse toolSpec)' do
        payload = render_payload(tools: { calculate: tool })
        tool_entry = payload[:tools].first
        expect(tool_entry).to have_key(:name)
        expect(tool_entry).to have_key(:description)
        expect(tool_entry).to have_key(:input_schema)
        expect(tool_entry).not_to have_key(:toolSpec)
      end
    end

    context 'with anthropic_beta configured' do
      it 'omits anthropic_beta when not configured' do
        payload = render_payload
        expect(payload).not_to have_key(:anthropic_beta)
      end

      it 'includes anthropic_beta array when configured' do
        p = build_protocol(anthropic_beta: 'interleaved-thinking-2025-05-14')
        payload = p.send(:render_payload, [], **base_args)
        expect(payload[:anthropic_beta]).to eq(['interleaved-thinking-2025-05-14'])
      end

      it 'accepts an array of betas' do
        p = build_protocol(anthropic_beta: %w[beta-a beta-b])
        payload = p.send(:render_payload, [], **base_args)
        expect(payload[:anthropic_beta]).to eq(%w[beta-a beta-b])
      end
    end

    context 'with context_management configured' do
      it 'omits context_management when not configured' do
        payload = render_payload
        expect(payload).not_to have_key(:context_management)
      end

      it 'includes context_management when configured' do
        value = { type: 'auto' }
        p = build_protocol(anthropic_context_management: value)
        payload = p.send(:render_payload, [], **base_args)
        expect(payload[:context_management]).to eq(value)
      end
    end
  end

  # ────────────────────────────────────────────────────────────────────────────
  # format_messages
  # ────────────────────────────────────────────────────────────────────────────

  describe '#format_messages' do
    it 'formats text messages as Anthropic type:text blocks' do
      msg = RubyLLM::Message.new(role: :user, content: 'Hello')
      result = protocol.send(:format_messages, [msg])

      expect(result.length).to eq(1)
      expect(result.first[:role]).to eq('user')
      content = result.first[:content]
      expect(content.first).to include(type: 'text', text: 'Hello')
    end

    it 'formats tool_use (assistant tool calls) as Anthropic type:tool_use blocks' do
      tool_call = RubyLLM::ToolCall.new(id: 'tc_1', name: 'calculate', arguments: { x: 1 })
      msg = RubyLLM::Message.new(role: :assistant, content: nil, tool_calls: { 'tc_1' => tool_call })

      result = protocol.send(:format_messages, [msg])
      block = result.first[:content].find { |b| b[:type] == 'tool_use' }

      expect(block).not_to be_nil
      expect(block[:id]).to eq('tc_1')
      expect(block[:name]).to eq('calculate')
      expect(block[:input]).to eq({ x: 1 })
    end

    it 'formats tool results as type:tool_result blocks' do
      msg = RubyLLM::Message.new(role: :tool, content: '42', tool_call_id: 'tc_1')

      result = protocol.send(:format_messages, [msg])
      # Tool results are grouped into a user message
      user_msg = result.find { |m| m[:role] == 'user' }
      expect(user_msg).not_to be_nil

      block = user_msg[:content].find { |b| b.is_a?(Hash) && b[:type] == 'tool_result' }
      expect(block).not_to be_nil
      expect(block[:tool_use_id]).to eq('tc_1')
      expect(block[:content].first[:text]).to eq('42')
    end

    it 'formats thinking blocks as Anthropic type:thinking blocks' do
      thinking = RubyLLM::Thinking.build(text: 'step 1', signature: 'sig123')
      msg = RubyLLM::Message.new(role: :assistant, content: 'answer', thinking: thinking)
      thinking_config = RubyLLM::Thinking::Config.new(budget: 1024)

      result = protocol.send(:format_messages, [msg], thinking: thinking_config)
      content = result.first[:content]

      thinking_block = content.find { |b| b[:type] == 'thinking' }
      expect(thinking_block).not_to be_nil
      expect(thinking_block[:thinking]).to eq('step 1')
      expect(thinking_block[:signature]).to eq('sig123')
    end

    it 'preserves cache_control on a content block passed via Content::Raw' do
      raw_block = { type: 'text', text: 'cached text', cache_control: { type: 'ephemeral' } }
      raw_content = RubyLLM::Content::Raw.new([raw_block])
      msg = RubyLLM::Message.new(role: :user, content: raw_content)

      result = protocol.send(:format_messages, [msg])
      block = result.first[:content].first

      expect(block[:cache_control]).to eq({ type: 'ephemeral' })
    end
  end

  # ────────────────────────────────────────────────────────────────────────────
  # parse_completion_body
  # ────────────────────────────────────────────────────────────────────────────

  describe '#parse_completion_body' do
    let(:chat_module) { RubyLLM::Protocols::BedrockInvokeModel::Chat }

    it 'extracts text content from Anthropic response' do
      data = {
        'content' => [{ 'type' => 'text', 'text' => 'Hello world' }],
        'usage' => { 'input_tokens' => 10, 'output_tokens' => 5 },
        'stop_reason' => 'end_turn',
        'model' => model.id
      }

      message = chat_module.parse_completion_body(data, raw: data)

      expect(message.content).to eq('Hello world')
      expect(message.finish_reason).to eq('end_turn')
      expect(message.model_id).to eq(model.id)
    end

    it 'extracts tool_use blocks into tool_calls' do
      data = {
        'content' => [
          { 'type' => 'tool_use', 'id' => 'tu_1', 'name' => 'search', 'input' => { 'q' => 'ruby' } }
        ],
        'usage' => { 'input_tokens' => 10, 'output_tokens' => 5 },
        'stop_reason' => 'tool_use'
      }

      message = chat_module.parse_completion_body(data, raw: data)

      expect(message.tool_calls).not_to be_nil
      tc = message.tool_calls['tu_1']
      expect(tc.id).to eq('tu_1')
      expect(tc.name).to eq('search')
      expect(tc.arguments).to eq({ 'q' => 'ruby' })
    end

    it 'normalizes cache tokens out of input_tokens' do
      data = {
        'content' => [{ 'type' => 'text', 'text' => 'hi' }],
        'usage' => {
          'input_tokens' => 100,
          'output_tokens' => 10,
          'cache_read_input_tokens' => 30,
          'cache_creation_input_tokens' => 20
        }
      }

      message = chat_module.parse_completion_body(data, raw: data)

      expect(message.input_tokens).to eq(50)
      expect(message.cached_tokens).to eq(30)
      expect(message.cache_creation_tokens).to eq(20)
    end

    it 'extracts thinking blocks' do
      data = {
        'content' => [
          { 'type' => 'thinking', 'thinking' => 'let me think', 'signature' => 'sig-abc' },
          { 'type' => 'text', 'text' => 'answer' }
        ],
        'usage' => { 'input_tokens' => 10, 'output_tokens' => 5 }
      }

      message = chat_module.parse_completion_body(data, raw: data)

      expect(message.thinking.text).to eq('let me think')
      expect(message.thinking.signature).to eq('sig-abc')
      expect(message.content).to eq('answer')
    end

    it 'preserves stop_reason as finish_reason' do
      data = {
        'content' => [{ 'type' => 'text', 'text' => 'ok' }],
        'usage' => {},
        'stop_reason' => 'max_tokens'
      }

      message = chat_module.parse_completion_body(data, raw: data)
      expect(message.finish_reason).to eq('max_tokens')
    end

    it 'parses all four usage fields from a final text response' do
      data = {
        'content' => [{ 'type' => 'text', 'text' => 'answer' }],
        'usage' => {
          'input_tokens' => 200,
          'output_tokens' => 50,
          'cache_read_input_tokens' => 80,
          'cache_creation_input_tokens' => 40
        },
        'stop_reason' => 'end_turn'
      }

      message = chat_module.parse_completion_body(data, raw: data)

      expect(message.input_tokens).to eq(80) # 200 - 80 - 40
      expect(message.output_tokens).to eq(50)
      expect(message.cached_tokens).to eq(80)
      expect(message.cache_creation_tokens).to eq(40)
    end

    it 'parses all four usage fields from an intermediate tool-call response' do
      data = {
        'content' => [
          { 'type' => 'tool_use', 'id' => 'tu_2', 'name' => 'lookup', 'input' => { 'id' => 7 } }
        ],
        'usage' => {
          'input_tokens' => 300,
          'output_tokens' => 20,
          'cache_read_input_tokens' => 150,
          'cache_creation_input_tokens' => 0
        },
        'stop_reason' => 'tool_use'
      }

      message = chat_module.parse_completion_body(data, raw: data)

      expect(message.input_tokens).to eq(150) # 300 - 150 - 0
      expect(message.output_tokens).to eq(20)
      expect(message.cached_tokens).to eq(150)
      expect(message.cache_creation_tokens).to eq(0)
      expect(message.finish_reason).to eq('tool_use')
    end
  end

  # ────────────────────────────────────────────────────────────────────────────
  # build_chunk (streaming)
  # ────────────────────────────────────────────────────────────────────────────

  describe '#build_chunk (streaming)' do
    let(:streaming) do
      Object.new.tap do |obj|
        obj.extend(RubyLLM::Protocols::BedrockInvokeModel::Streaming)
        obj.instance_variable_set(:@model, instance_double(RubyLLM::Model::Info, id: model.id))
      end
    end

    it 'extracts input_tokens from message_start' do
      event = {
        'type' => 'message_start',
        'message' => {
          'model' => model.id,
          'usage' => { 'input_tokens' => 42 }
        }
      }

      chunk = streaming.send(:build_chunk, event)
      expect(chunk.input_tokens).to eq(42)
      expect(chunk.model_id).to eq(model.id)
    end

    it 'returns a tool_call start chunk for content_block_start with tool_use' do
      event = {
        'type' => 'content_block_start',
        'index' => 0,
        'content_block' => { 'type' => 'tool_use', 'id' => 'tu_x', 'name' => 'my_tool' }
      }

      chunk = streaming.send(:build_chunk, event)
      expect(chunk.tool_calls).not_to be_nil
      # The stream key is the block index so the StreamAccumulator can match deltas by index.
      tc = chunk.tool_calls[0]
      expect(tc.id).to eq('tu_x')
      expect(tc.name).to eq('my_tool')
    end

    it 'returns nil for content_block_start with non-tool_use type' do
      event = {
        'type' => 'content_block_start',
        'index' => 0,
        'content_block' => { 'type' => 'text', 'text' => '' }
      }

      chunk = streaming.send(:build_chunk, event)
      expect(chunk).to be_nil
    end

    it 'extracts text from text_delta content_block_delta' do
      event = {
        'type' => 'content_block_delta',
        'index' => 0,
        'delta' => { 'type' => 'text_delta', 'text' => 'Hello' }
      }

      chunk = streaming.send(:build_chunk, event)
      expect(chunk.content).to eq('Hello')
    end

    it 'extracts partial_json from input_json_delta for tool call accumulation' do
      event = {
        'type' => 'content_block_delta',
        'index' => 1,
        'delta' => { 'type' => 'input_json_delta', 'partial_json' => '{"q":' }
      }

      chunk = streaming.send(:build_chunk, event)
      expect(chunk.tool_calls).not_to be_nil
      tc = chunk.tool_calls[1]
      expect(tc.arguments).to eq('{"q":')
    end

    it 'extracts thinking text from thinking_delta' do
      event = {
        'type' => 'content_block_delta',
        'index' => 0,
        'delta' => { 'type' => 'thinking_delta', 'thinking' => 'pondering...' }
      }

      chunk = streaming.send(:build_chunk, event)
      expect(chunk.thinking.text).to eq('pondering...')
    end

    it 'extracts signature from signature_delta' do
      event = {
        'type' => 'content_block_delta',
        'index' => 0,
        'delta' => { 'type' => 'signature_delta', 'signature' => 'sig-xyz' }
      }

      chunk = streaming.send(:build_chunk, event)
      expect(chunk.thinking.signature).to eq('sig-xyz')
    end

    it 'extracts stop_reason and output_tokens from message_delta' do
      event = {
        'type' => 'message_delta',
        'delta' => { 'stop_reason' => 'end_turn' },
        'usage' => { 'output_tokens' => 77 }
      }

      chunk = streaming.send(:build_chunk, event)
      expect(chunk.finish_reason).to eq('end_turn')
      expect(chunk.output_tokens).to eq(77)
    end

    it 'returns nil for content_block_stop' do
      event = { 'type' => 'content_block_stop', 'index' => 0 }
      expect(streaming.send(:build_chunk, event)).to be_nil
    end

    it 'returns nil for message_stop' do
      event = { 'type' => 'message_stop' }
      expect(streaming.send(:build_chunk, event)).to be_nil
    end

    it 'accumulates content_block_deltas into a complete message via StreamAccumulator' do
      accumulator = RubyLLM::StreamAccumulator.new

      accumulator.add(streaming.send(:build_chunk, {
                                       'type' => 'message_start',
                                       'message' => { 'model' => model.id, 'usage' => { 'input_tokens' => 10 } }
                                     }))

      accumulator.add(streaming.send(:build_chunk, {
                                       'type' => 'content_block_delta',
                                       'index' => 0,
                                       'delta' => { 'type' => 'text_delta', 'text' => 'Hello ' }
                                     }))

      accumulator.add(streaming.send(:build_chunk, {
                                       'type' => 'content_block_delta',
                                       'index' => 0,
                                       'delta' => { 'type' => 'text_delta', 'text' => 'world' }
                                     }))

      accumulator.add(streaming.send(:build_chunk, {
                                       'type' => 'message_delta',
                                       'delta' => { 'stop_reason' => 'end_turn' },
                                       'usage' => { 'output_tokens' => 5 }
                                     }))

      message = accumulator.to_message(nil)
      expect(message.content).to eq('Hello world')
      expect(message.input_tokens).to eq(10)
      expect(message.output_tokens).to eq(5)
      expect(message.finish_reason).to eq('end_turn')
    end

    it 'accumulates tool call start + input_json_delta into a complete tool call' do
      accumulator = RubyLLM::StreamAccumulator.new

      accumulator.add(streaming.send(:build_chunk, {
                                       'type' => 'content_block_start',
                                       'index' => 0,
                                       'content_block' => { 'type' => 'tool_use', 'id' => 'tu_1', 'name' => 'calc' }
                                     }))

      accumulator.add(streaming.send(:build_chunk, {
                                       'type' => 'content_block_delta',
                                       'index' => 0,
                                       'delta' => { 'type' => 'input_json_delta', 'partial_json' => '{"x":' }
                                     }))

      accumulator.add(streaming.send(:build_chunk, {
                                       'type' => 'content_block_delta',
                                       'index' => 0,
                                       'delta' => { 'type' => 'input_json_delta', 'partial_json' => '42}' }
                                     }))

      message = accumulator.to_message(nil)
      expect(message.tool_calls).not_to be_nil
      tc = message.tool_calls['tu_1']
      expect(tc.name).to eq('calc')
      expect(tc.arguments).to eq({ 'x' => 42 })
    end
  end

  # ────────────────────────────────────────────────────────────────────────────
  # URL methods
  # ────────────────────────────────────────────────────────────────────────────

  describe 'URL methods' do
    it 'completion_url ends with /invoke' do
      url = protocol.send(:completion_url)
      expect(url).to end_with('/invoke')
      expect(url).to include('invoke')
      expect(url).not_to include('converse')
    end

    it 'stream_url ends with /invoke-with-response-stream' do
      url = protocol.send(:stream_url)
      expect(url).to end_with('/invoke-with-response-stream')
      expect(url).not_to include('converse')
    end

    it 'percent-encodes "/" in application inference profile ARNs for completion_url' do
      allow(model).to receive(:id).and_return(
        'arn:aws:bedrock:us-west-2:123:application-inference-profile/p'
      )
      url = protocol.send(:completion_url)
      expect(url).to eq(
        '/model/arn:aws:bedrock:us-west-2:123:application-inference-profile%2Fp/invoke'
      )
    end
  end

  # ────────────────────────────────────────────────────────────────────────────
  # Provider selection
  # ────────────────────────────────────────────────────────────────────────────

  describe 'provider protocol selection' do
    def build_provider(config_overrides = {})
      config = build_config(config_overrides)
      RubyLLM::Providers::Bedrock.new(config)
    end

    it 'uses Converse when bedrock_use_invoke_model is false (default)' do
      provider = build_provider
      protocol_class = provider.send(:protocol_for, model)
      expect(protocol_class).to eq(RubyLLM::Protocols::Converse)
    end

    it 'uses BedrockInvokeModel when bedrock_use_invoke_model is true' do
      provider = build_provider(bedrock_use_invoke_model: true)
      protocol_class = provider.send(:protocol_for, model)
      expect(protocol_class).to eq(described_class)
    end

    it 'uses Converse when bedrock_use_invoke_model is nil' do
      provider = build_provider(bedrock_use_invoke_model: nil)
      protocol_class = provider.send(:protocol_for, model)
      expect(protocol_class).to eq(RubyLLM::Protocols::Converse)
    end

    it 'registers bedrock_use_invoke_model as a configuration option' do
      expect(RubyLLM::Configuration.options).to include(:bedrock_use_invoke_model)
    end

    it 'registers anthropic_beta as a configuration option' do
      expect(RubyLLM::Configuration.options).to include(:anthropic_beta)
    end

    it 'registers anthropic_context_management as a configuration option' do
      expect(RubyLLM::Configuration.options).to include(:anthropic_context_management)
    end

    it 'BedrockInvokeModel completion URL hits /invoke not /converse' do
      provider = build_provider(bedrock_use_invoke_model: true)
      protocol_class = provider.send(:protocol_for, model)
      instance = protocol_class.allocate
      instance.instance_variable_set(:@model, model)
      expect(instance.send(:completion_url)).to include('/invoke')
      expect(instance.send(:completion_url)).not_to include('converse')
    end

    it 'Converse completion URL hits /converse not /invoke' do
      provider = build_provider
      protocol_class = provider.send(:protocol_for, model)
      instance = protocol_class.allocate
      instance.instance_variable_set(:@model, model)
      expect(instance.send(:completion_url)).to include('/converse')
      expect(instance.send(:completion_url)).not_to include('invoke')
    end
  end
end
