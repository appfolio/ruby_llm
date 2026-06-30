# frozen_string_literal: true

require 'spec_helper'

RSpec.describe RubyLLM::Protocols::BedrockInvokeModel do
  # ---------------------------------------------------------------------------
  # Helpers
  # ---------------------------------------------------------------------------

  def build_model(id, max_tokens: 4096)
    instance_double(
      RubyLLM::Model::Info,
      id: id,
      max_tokens: max_tokens,
      metadata: {}
    )
  end

  def build_config(overrides = {})
    cfg = RubyLLM::Configuration.new
    cfg.bedrock_api_key = 'test-key'
    cfg.bedrock_secret_key = 'test-secret'
    cfg.bedrock_region = 'us-east-1'
    overrides.each { |k, v| cfg.public_send(:"#{k}=", v) }
    cfg
  end

  # Returns a BedrockInvokeModel instance with @model and @config set.
  def make_instance(model_id: 'anthropic.claude-haiku-4-5-20251001-v1:0',
                    max_tokens: 4096,
                    config_overrides: {})
    config = build_config(config_overrides)
    model  = build_model(model_id, max_tokens: max_tokens)
    described_class.allocate.tap do |obj|
      obj.instance_variable_set(:@model, model)
      obj.instance_variable_set(:@config, config)
    end
  end

  def render_payload(messages = [], **opts)
    model_id = opts.fetch(:model_id, 'anthropic.claude-haiku-4-5-20251001-v1:0')
    max_tokens = opts.fetch(:max_tokens, 4096)
    tools = opts.fetch(:tools, {})
    temperature = opts.fetch(:temperature, nil)
    thinking = opts.fetch(:thinking, nil)
    config_overrides = opts.fetch(:config_overrides, {})

    inst = make_instance(model_id: model_id, max_tokens: max_tokens, config_overrides: config_overrides)
    model = inst.instance_variable_get(:@model)
    inst.send(:render_payload, messages,
              tools: tools, temperature: temperature, model: model, thinking: thinking)
  end

  # ---------------------------------------------------------------------------
  # escape_model_id
  # ---------------------------------------------------------------------------

  describe 'Chat#escape_model_id' do
    subject(:chat) { described_class::Chat }

    it 'leaves a plain model id unchanged' do
      expect(chat.escape_model_id('anthropic.claude-haiku-4-5-20251001-v1:0'))
        .to eq('anthropic.claude-haiku-4-5-20251001-v1:0')
    end

    it 'percent-encodes slashes in application-inference-profile ARNs' do
      arn = 'arn:aws:bedrock:us-west-2:123456789012:application-inference-profile/abc123'
      encoded = chat.escape_model_id(arn)
      expect(encoded).not_to include('/')
      expect(encoded).to include('%2F')
    end

    it 'generates /invoke URL with encoded ARN' do
      arn = 'arn:aws:bedrock:us-west-2:123456789012:application-inference-profile/abc123'
      inst = make_instance(model_id: arn)
      url = inst.send(:completion_url)
      expect(url).to start_with('/model/')
      expect(url).to end_with('/invoke')
      expect(url).not_to include('application-inference-profile/')
    end

    it 'generates /invoke-with-response-stream URL with encoded ARN' do
      arn = 'arn:aws:bedrock:us-west-2:123456789012:application-inference-profile/abc123'
      inst = make_instance(model_id: arn)
      url = inst.send(:stream_url)
      expect(url).to end_with('/invoke-with-response-stream')
      expect(url).not_to include('application-inference-profile/')
    end
  end

  # ---------------------------------------------------------------------------
  # render_payload
  # ---------------------------------------------------------------------------

  describe 'Chat#render_payload' do
    it 'includes anthropic_version' do
      expect(render_payload[:anthropic_version]).to eq('bedrock-2023-05-31')
    end

    it 'places max_tokens at the top level' do
      expect(render_payload(max_tokens: 8192)[:max_tokens]).to eq(8192)
    end

    it 'does NOT include a model field in the body' do
      expect(render_payload).not_to have_key(:model)
      expect(render_payload).not_to have_key('model')
    end

    it 'defaults max_tokens to 4096 when model.max_tokens is nil' do
      expect(render_payload(max_tokens: nil)[:max_tokens]).to eq(4096)
    end

    it 'includes temperature when provided' do
      expect(render_payload(temperature: 0.7)[:temperature]).to eq(0.7)
    end

    it 'omits temperature when nil' do
      expect(render_payload(temperature: nil)).not_to have_key(:temperature)
    end

    it 'formats messages in Anthropic shape (text blocks)' do
      msg = RubyLLM::Message.new(role: :user, content: 'Hello')
      result = render_payload([msg])
      expect(result[:messages].first[:role]).to eq('user')
      expect(result[:messages].first[:content].first[:type]).to eq('text')
      expect(result[:messages].first[:content].first[:text]).to eq('Hello')
    end

    it 'formats system content as top-level :system array' do
      sys = RubyLLM::Message.new(role: :system, content: 'You are helpful')
      result = render_payload([sys])
      expect(result[:system]).to be_an(Array)
      expect(result[:system].first[:type]).to eq('text')
      expect(result[:system].first[:text]).to eq('You are helpful')
      expect(result[:messages]).to be_empty
    end

    it 'formats tool definitions in Anthropic shape (input_schema)' do
      tool = instance_double(
        RubyLLM::Tool,
        name: 'my_tool',
        description: 'does stuff',
        parameters: {},
        params_schema: { 'type' => 'object', 'properties' => {}, 'required' => [] },
        provider_params: {}
      )
      result = render_payload(tools: { 'my_tool' => tool })
      expect(result[:tools]).not_to be_nil
      expect(result[:tools].first[:name]).to eq('my_tool')
      expect(result[:tools].first[:input_schema]).to be_a(Hash)
    end

    it 'uses Converse-style toolSpec shape is NOT present (Anthropic native shape)' do
      tool = instance_double(
        RubyLLM::Tool,
        name: 'my_tool',
        description: 'does stuff',
        parameters: {},
        params_schema: nil,
        provider_params: {}
      )
      result = render_payload(tools: { 'my_tool' => tool })
      expect(result[:tools].first).not_to have_key(:toolSpec)
    end

    it 'includes anthropic_beta array when configured' do
      result = render_payload(config_overrides: { anthropic_beta: ['interleaved-thinking-2025-05-14'] })
      expect(result[:anthropic_beta]).to eq(['interleaved-thinking-2025-05-14'])
    end

    it 'wraps a scalar anthropic_beta in an array' do
      result = render_payload(config_overrides: { anthropic_beta: 'prompt-caching-2024-07-31' })
      expect(result[:anthropic_beta]).to eq(['prompt-caching-2024-07-31'])
    end

    it 'includes context_management when configured' do
      result = render_payload(config_overrides: { anthropic_context_management: { type: 'auto' } })
      expect(result[:context_management]).to eq({ type: 'auto' })
    end

    it 'omits anthropic_beta when not configured' do
      result = render_payload
      expect(result).not_to have_key(:anthropic_beta)
    end

    it 'omits context_management when not configured' do
      result = render_payload
      expect(result).not_to have_key(:context_management)
    end
  end

  # ---------------------------------------------------------------------------
  # format_messages
  # ---------------------------------------------------------------------------

  describe 'Chat#format_messages' do
    subject(:chat) { described_class::Chat }

    it 'formats text content as {type: text, text: ...} blocks' do
      msg = RubyLLM::Message.new(role: :user, content: 'hi')
      result = chat.format_messages([msg])
      expect(result.first[:content].first).to eq({ type: 'text', text: 'hi' })
    end

    it 'formats tool_use blocks in Anthropic shape' do
      tool_call = RubyLLM::ToolCall.new(id: 'call_1', name: 'my_tool', arguments: { x: 1 })
      msg = RubyLLM::Message.new(role: :assistant, content: '', tool_calls: { 'call_1' => tool_call })
      result = chat.format_messages([msg])
      block = result.first[:content].find { |b| b[:type] == 'tool_use' }
      expect(block[:type]).to eq('tool_use')
      expect(block[:id]).to eq('call_1')
      expect(block[:name]).to eq('my_tool')
      expect(block[:input]).to eq({ x: 1 })
    end

    it 'formats tool_result blocks in Anthropic shape' do
      msg = RubyLLM::Message.new(role: :tool, content: 'result text', tool_call_id: 'call_1')
      result = chat.format_messages([msg])
      content = result.first[:content]
      block = content.find { |b| b.is_a?(Hash) && b[:type] == 'tool_result' }
      expect(block[:type]).to eq('tool_result')
      expect(block[:tool_use_id]).to eq('call_1')
    end

    it 'formats thinking blocks in Anthropic shape when role is assistant' do
      thinking = RubyLLM::Thinking.build(text: 'my thought', signature: 'sig')
      msg = RubyLLM::Message.new(role: :assistant, content: 'reply', thinking: thinking)
      result = chat.format_messages([msg])
      thinking_block = result.first[:content].find { |b| b[:type] == 'thinking' }
      expect(thinking_block[:type]).to eq('thinking')
      expect(thinking_block[:thinking]).to eq('my thought')
      expect(thinking_block[:signature]).to eq('sig')
    end

    it 'preserves cache_control when block already carries one' do
      raw_block = { 'type' => 'text', 'text' => 'cached', 'cache_control' => { 'type' => 'ephemeral' } }
      raw_content = RubyLLM::Content::Raw.new([raw_block])
      msg = RubyLLM::Message.new(role: :user, content: raw_content)
      result = chat.format_messages([msg])
      expect(result.first[:content].first['cache_control']).to eq({ 'type' => 'ephemeral' })
    end

    it 'does not inject cache_control when block has none' do
      msg = RubyLLM::Message.new(role: :user, content: 'plain text')
      result = chat.format_messages([msg])
      block = result.first[:content].first
      expect(block).not_to have_key(:cache_control)
      expect(block).not_to have_key('cache_control')
    end
  end

  # ---------------------------------------------------------------------------
  # parse_completion_body
  # ---------------------------------------------------------------------------

  describe 'Chat#parse_completion_body' do
    subject(:chat) { described_class::Chat }

    let(:basic_response) do
      {
        'id' => 'msg_01',
        'type' => 'message',
        'model' => 'anthropic.claude-haiku-4-5-20251001-v1:0',
        'stop_reason' => 'end_turn',
        'content' => [{ 'type' => 'text', 'text' => 'Hello!' }],
        'usage' => {
          'input_tokens' => 20,
          'output_tokens' => 5,
          'cache_read_input_tokens' => 0,
          'cache_creation_input_tokens' => 0
        }
      }
    end

    it 'extracts text content' do
      msg = chat.parse_completion_body(basic_response, raw: nil)
      expect(msg.content).to eq('Hello!')
    end

    it 'extracts stop_reason as finish_reason' do
      msg = chat.parse_completion_body(basic_response, raw: nil)
      expect(msg.finish_reason).to eq('end_turn')
    end

    it 'extracts model_id from response' do
      msg = chat.parse_completion_body(basic_response, raw: nil)
      expect(msg.model_id).to eq('anthropic.claude-haiku-4-5-20251001-v1:0')
    end

    it 'extracts output_tokens from usage' do
      msg = chat.parse_completion_body(basic_response, raw: nil)
      expect(msg.output_tokens).to eq(5)
    end

    it 'extracts input_tokens net of cache tokens' do
      data = basic_response.merge(
        'usage' => {
          'input_tokens' => 100,
          'output_tokens' => 5,
          'cache_read_input_tokens' => 40,
          'cache_creation_input_tokens' => 10
        }
      )
      msg = chat.parse_completion_body(data, raw: nil)
      expect(msg.input_tokens).to eq(50)
      expect(msg.cached_tokens).to eq(40)
      expect(msg.cache_creation_tokens).to eq(10)
    end

    it 'extracts tool_use blocks as tool_calls' do
      data = basic_response.merge(
        'stop_reason' => 'tool_use',
        'content' => [
          {
            'type' => 'tool_use',
            'id' => 'call_abc',
            'name' => 'my_tool',
            'input' => { 'arg' => 'val' }
          }
        ]
      )
      msg = chat.parse_completion_body(data, raw: nil)
      expect(msg.tool_calls).not_to be_nil
      tc = msg.tool_calls['call_abc']
      expect(tc.name).to eq('my_tool')
      expect(tc.arguments).to eq({ 'arg' => 'val' })
    end

    it 'extracts thinking blocks' do
      data = basic_response.merge(
        'content' => [
          { 'type' => 'thinking', 'thinking' => 'I am thinking', 'signature' => 'sig123' },
          { 'type' => 'text', 'text' => 'Done' }
        ]
      )
      msg = chat.parse_completion_body(data, raw: nil)
      expect(msg.thinking.text).to eq('I am thinking')
      expect(msg.thinking.signature).to eq('sig123')
    end

    it 'returns nil for empty data' do
      expect(chat.parse_completion_body(nil, raw: nil)).to be_nil
      expect(chat.parse_completion_body({}, raw: nil)).to be_nil
    end
  end

  # ---------------------------------------------------------------------------
  # build_chunk (streaming)
  # ---------------------------------------------------------------------------

  describe 'Streaming#build_chunk' do
    let(:streaming) do
      described_class.allocate.tap do |obj|
        obj.instance_variable_set(:@model, build_model('anthropic.claude-haiku-4-5-20251001-v1:0'))
        obj.instance_variable_set(:@config, build_config)
      end
    end

    it 'extracts input_tokens from message_start' do
      event = {
        'type' => 'message_start',
        'message' => {
          'id' => 'msg_01',
          'model' => 'anthropic.claude-sonnet-4-6',
          'usage' => { 'input_tokens' => 42 }
        }
      }
      chunk = streaming.send(:build_chunk, event)
      expect(chunk.input_tokens).to eq(42)
    end

    it 'extracts model_id from message_start' do
      event = {
        'type' => 'message_start',
        'message' => {
          'model' => 'anthropic.claude-sonnet-4-6',
          'usage' => { 'input_tokens' => 10 }
        }
      }
      chunk = streaming.send(:build_chunk, event)
      expect(chunk.model_id).to eq('anthropic.claude-sonnet-4-6')
    end

    it 'extracts text from content_block_delta text_delta' do
      event = {
        'type' => 'content_block_delta',
        'index' => 0,
        'delta' => { 'type' => 'text_delta', 'text' => 'Hello' }
      }
      chunk = streaming.send(:build_chunk, event)
      expect(chunk.content).to eq('Hello')
    end

    it 'extracts partial_json from content_block_delta input_json_delta' do
      event = {
        'type' => 'content_block_delta',
        'index' => 1,
        'delta' => { 'type' => 'input_json_delta', 'partial_json' => '{"key":' }
      }
      chunk = streaming.send(:build_chunk, event)
      expect(chunk.tool_calls).not_to be_nil
      expect(chunk.tool_calls[nil].arguments).to eq('{"key":')
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

    it 'extracts thinking signature from signature_delta' do
      event = {
        'type' => 'content_block_delta',
        'index' => 0,
        'delta' => { 'type' => 'signature_delta', 'signature' => 'sig-abc' }
      }
      chunk = streaming.send(:build_chunk, event)
      expect(chunk.thinking.signature).to eq('sig-abc')
    end

    it 'extracts output_tokens and stop_reason from message_delta' do
      event = {
        'type' => 'message_delta',
        'delta' => { 'stop_reason' => 'end_turn' },
        'usage' => { 'output_tokens' => 17 }
      }
      chunk = streaming.send(:build_chunk, event)
      expect(chunk.output_tokens).to eq(17)
      expect(chunk.finish_reason).to eq('end_turn')
    end

    it 'returns a chunk for message_stop without error' do
      event = { 'type' => 'message_stop' }
      expect { streaming.send(:build_chunk, event) }.not_to raise_error
    end

    it 'extracts tool_call from content_block_start tool_use' do
      event = {
        'type' => 'content_block_start',
        'index' => 0,
        'content_block' => {
          'type' => 'tool_use',
          'id' => 'call_xyz',
          'name' => 'search'
        }
      }
      chunk = streaming.send(:build_chunk, event)
      expect(chunk.tool_calls).not_to be_nil
      tc = chunk.tool_calls['call_xyz']
      expect(tc.name).to eq('search')
    end

    it 'accumulates streaming chunks into a final message' do
      accumulator = RubyLLM::StreamAccumulator.new

      events = [
        { 'type' => 'message_start', 'message' => { 'model' => 'test-model', 'usage' => { 'input_tokens' => 5 } } },
        { 'type' => 'content_block_delta', 'index' => 0,
          'delta' => { 'type' => 'text_delta', 'text' => 'Hello' } },
        { 'type' => 'content_block_delta', 'index' => 0,
          'delta' => { 'type' => 'text_delta', 'text' => ' world' } },
        { 'type' => 'message_delta', 'delta' => { 'stop_reason' => 'end_turn' },
          'usage' => { 'output_tokens' => 2 } }
      ]

      events.each { |e| accumulator.add(streaming.send(:build_chunk, e)) }
      message = accumulator.to_message(nil)

      expect(message.content).to eq('Hello world')
      expect(message.output_tokens).to eq(2)
    end
  end

  # ---------------------------------------------------------------------------
  # Model coverage — two different Claude models + one ARN
  # ---------------------------------------------------------------------------

  describe 'URL generation for multiple model ids' do
    [
      'anthropic.claude-haiku-4-5-20251001-v1:0',
      'anthropic.claude-sonnet-4-5-20250929-v1:0'
    ].each do |model_id|
      it "generates /invoke URL for #{model_id}" do
        inst = make_instance(model_id: model_id)
        expect(inst.send(:completion_url)).to eq("/model/#{model_id}/invoke")
      end
    end

    it 'percent-encodes slashes in ARN model ids in invoke URL' do
      arn = 'arn:aws:bedrock:us-west-2:999999999999:application-inference-profile/my-profile'
      inst = make_instance(model_id: arn)
      url = inst.send(:completion_url)
      expect(url).to include('%2F')
      expect(url).not_to include('application-inference-profile/')
    end

    it 'percent-encodes slashes in ARN model ids in invoke-with-response-stream URL' do
      arn = 'arn:aws:bedrock:us-west-2:999999999999:application-inference-profile/my-profile'
      inst = make_instance(model_id: arn)
      url = inst.send(:stream_url)
      expect(url).to include('%2F')
      expect(url).not_to include('application-inference-profile/')
    end
  end

  # ---------------------------------------------------------------------------
  # Coexistence / selection — bedrock_use_invoke_model
  # ---------------------------------------------------------------------------

  describe 'Providers::Bedrock protocol selection' do # rubocop:disable RSpec/MultipleMemoizedHelpers
    def build_bedrock(use_invoke_model: false)
      config = RubyLLM::Configuration.new
      config.bedrock_api_key = 'k'
      config.bedrock_secret_key = 's'
      config.bedrock_region = 'us-east-1'
      config.bedrock_use_invoke_model = use_invoke_model
      RubyLLM::Providers::Bedrock.new(config)
    end

    def model_double(id, metadata: {})
      instance_double(RubyLLM::Model::Info, id: id, max_tokens: 4096, metadata: metadata)
    end

    let(:haiku_id)       { 'anthropic.claude-haiku-4-5-20251001-v1:0' }
    let(:sonnet_id)      { 'anthropic.claude-sonnet-4-6-20250514-v1:0' }
    let(:nova_id)        { 'amazon.nova-lite-v1:0' }
    let(:arn_id)         { 'arn:aws:bedrock:us-west-2:123456789012:application-inference-profile/sonnet46' }
    let(:arn_anthropic)  { model_double(arn_id, metadata: { provider_name: 'Anthropic' }) }
    let(:arn_no_vendor)  { model_double(arn_id, metadata: {}) }

    context 'with bedrock_use_invoke_model: false (default)' do # rubocop:disable RSpec/MultipleMemoizedHelpers
      let(:provider) { build_bedrock(use_invoke_model: false) }

      it 'routes Anthropic models to Converse' do
        protocol = provider.protocol_for(model_double(haiku_id))
        expect(protocol).to be(RubyLLM::Protocols::Converse)
      end

      it 'routes non-Anthropic models to Converse' do
        protocol = provider.protocol_for(model_double(nova_id))
        expect(protocol).to be(RubyLLM::Protocols::Converse)
      end
    end

    context 'with bedrock_use_invoke_model: true' do # rubocop:disable RSpec/MultipleMemoizedHelpers
      let(:provider) { build_bedrock(use_invoke_model: true) }

      it 'routes Anthropic models to BedrockInvokeModel' do
        protocol = provider.protocol_for(model_double(haiku_id))
        expect(protocol).to be(described_class)
      end

      it 'routes non-Anthropic models to Converse regardless of flag' do
        protocol = provider.protocol_for(model_double(nova_id))
        expect(protocol).to be(RubyLLM::Protocols::Converse)
      end

      it 'routes ARN model ids with Anthropic provider_name to BedrockInvokeModel' do
        protocol = provider.protocol_for(arn_anthropic)
        expect(protocol).to be(described_class)
      end

      it 'routes ARN model ids without provider_name to Converse and logs a warning' do
        allow(RubyLLM.logger).to receive(:warn)
        protocol = provider.protocol_for(arn_no_vendor)
        expect(protocol).to be(RubyLLM::Protocols::Converse)
        expect(RubyLLM.logger).to have_received(:warn).with(/cannot verify.*Anthropic-backed/)
      end
    end

    context 'with bedrock_use_invoke_model: [array of model ids]' do # rubocop:disable RSpec/MultipleMemoizedHelpers
      let(:provider) { build_bedrock(use_invoke_model: [sonnet_id]) }

      it 'routes model A (in list) to BedrockInvokeModel' do
        protocol = provider.protocol_for(model_double(sonnet_id))
        expect(protocol).to be(described_class)
      end

      it 'routes model B (not in list) to Converse' do
        protocol = provider.protocol_for(model_double(haiku_id))
        expect(protocol).to be(RubyLLM::Protocols::Converse)
      end
    end

    context 'with bedrock_use_invoke_model: callable predicate' do # rubocop:disable RSpec/MultipleMemoizedHelpers
      let(:selector) { ->(model) { model.id == sonnet_id } }
      let(:provider) { build_bedrock(use_invoke_model: selector) }

      it 'routes model matching predicate to BedrockInvokeModel' do
        protocol = provider.protocol_for(model_double(sonnet_id))
        expect(protocol).to be(described_class)
      end

      it 'routes model not matching predicate to Converse' do
        protocol = provider.protocol_for(model_double(haiku_id))
        expect(protocol).to be(RubyLLM::Protocols::Converse)
      end
    end
  end

  # ---------------------------------------------------------------------------
  # render_payload — schema/citations warnings
  # ---------------------------------------------------------------------------

  describe 'Chat#render_payload warnings' do
    it 'logs a warning when schema is passed' do
      inst = make_instance
      model = inst.instance_variable_get(:@model)
      schema = { name: 'out', schema: { type: 'object' } }
      allow(RubyLLM.logger).to receive(:warn)
      inst.send(:render_payload, [], tools: {}, temperature: nil, model: model, schema: schema)
      expect(RubyLLM.logger).to have_received(:warn).with(/structured output.*schema.*BedrockInvokeModel/i)
    end

    it 'logs a warning when citations is true' do
      inst = make_instance
      model = inst.instance_variable_get(:@model)
      allow(RubyLLM.logger).to receive(:warn)
      inst.send(:render_payload, [], tools: {}, temperature: nil, model: model, citations: true)
      expect(RubyLLM.logger).to have_received(:warn).with(/citations.*BedrockInvokeModel/i)
    end

    it 'does not warn when neither schema nor citations are set' do
      inst = make_instance
      model = inst.instance_variable_get(:@model)
      allow(RubyLLM.logger).to receive(:warn)
      inst.send(:render_payload, [], tools: {}, temperature: nil, model: model)
      expect(RubyLLM.logger).not_to have_received(:warn)
    end
  end

  # ---------------------------------------------------------------------------
  # Large-file upload overrides
  # ---------------------------------------------------------------------------

  describe 'Chat large-file upload overrides' do
    subject(:chat) { described_class::Chat }

    it 'supports provider file references' do
      inst = make_instance
      expect(inst.send(:supports_provider_file_references?)).to be(true)
    end

    it 'uses the same 4.5 MB inline threshold as Converse' do
      inst = make_instance
      expect(inst.send(:default_large_file_upload_threshold))
        .to eq(RubyLLM::Protocols::Converse::Chat::BEDROCK_INLINE_DOCUMENT_LIMIT)
    end

    it 'marks pdf attachments as uploadable' do
      pdf = instance_double(RubyLLM::Attachment, pdf?: true, document?: false, text?: false)
      inst = make_instance
      expect(inst.send(:provider_file_attachable?, pdf)).to be(true)
    end

    it 'marks document attachments as uploadable' do
      doc = instance_double(RubyLLM::Attachment, pdf?: false, document?: true, text?: false)
      inst = make_instance
      expect(inst.send(:provider_file_attachable?, doc)).to be(true)
    end

    it 'marks text attachments as uploadable' do
      txt = instance_double(RubyLLM::Attachment, pdf?: false, document?: false, text?: true)
      inst = make_instance
      expect(inst.send(:provider_file_attachable?, txt)).to be(true)
    end

    it 'does not mark image attachments as uploadable' do
      img = instance_double(RubyLLM::Attachment, pdf?: false, document?: false, text?: false)
      inst = make_instance
      expect(inst.send(:provider_file_attachable?, img)).to be(false)
    end
  end

  # ---------------------------------------------------------------------------
  # URL image source rejection
  # ---------------------------------------------------------------------------

  describe 'Chat#format_image_attachment' do
    subject(:chat) { described_class::Chat }

    it 'raises UnsupportedAttachmentError for URL-sourced images' do
      attachment = instance_double(
        RubyLLM::Attachment,
        url?: true,
        source: URI.parse('https://example.com/photo.jpg'),
        mime_type: 'image/jpeg',
        encoded: nil
      )
      expect { chat.format_image_attachment(attachment) }
        .to raise_error(RubyLLM::UnsupportedAttachmentError, /Bedrock InvokeModel.*URL image/)
    end

    it 'returns a base64 block for non-URL images' do
      attachment = instance_double(
        RubyLLM::Attachment,
        url?: false,
        mime_type: 'image/jpeg',
        encoded: 'base64data'
      )
      result = chat.format_image_attachment(attachment)
      expect(result[:source][:type]).to eq('base64')
      expect(result[:source][:data]).to eq('base64data')
    end
  end
end
