# frozen_string_literal: true

require 'spec_helper'

RSpec.describe RubyLLM::Protocols::Converse::Chat do
  describe '.parse_completion_response' do
    it 'normalizes cache read and write tokens out of input tokens' do
      response_body = {
        'modelId' => 'anthropic.claude-sonnet-4-5-20250929-v1:0',
        'output' => {
          'message' => {
            'content' => [{ 'text' => 'Hi!' }]
          }
        },
        'usage' => {
          'inputTokens' => 100,
          'outputTokens' => 5,
          'cacheReadInputTokens' => 40,
          'cacheWriteInputTokens' => 10
        }
      }

      response = instance_double(Faraday::Response, body: response_body)
      message = described_class.parse_completion_response(response)

      expect(message.input_tokens).to eq(50)
      expect(message.output_tokens).to eq(5)
      expect(message.cached_tokens).to eq(40)
      expect(message.cache_creation_tokens).to eq(10)
    end

    it 'preserves raw stopReason as finish_reason' do
      response_body = {
        'modelId' => 'amazon.nova-lite-v1:0',
        'output' => {
          'message' => {
            'content' => [{ 'text' => 'Hi!' }]
          }
        },
        'stopReason' => 'guardrail_intervened',
        'usage' => {}
      }

      response = instance_double(Faraday::Response, body: response_body)
      message = described_class.parse_completion_response(response)

      expect(message.finish_reason).to eq('guardrail_intervened')
    end

    it 'extracts thinking tokens from top-level reasoningTokens' do
      response_body = {
        'output' => {
          'message' => {
            'content' => [{ 'text' => 'Hi!' }]
          }
        },
        'usage' => {
          'inputTokens' => 10,
          'outputTokens' => 5,
          'reasoningTokens' => 7
        }
      }

      response = instance_double(Faraday::Response, body: response_body)
      message = described_class.parse_completion_response(response)

      expect(message.thinking_tokens).to eq(7)
    end

    it 'extracts thinking tokens from outputTokensDetails reasoningTokens' do
      response_body = {
        'output' => {
          'message' => {
            'content' => [{ 'text' => 'Hi!' }]
          }
        },
        'usage' => {
          'inputTokens' => 10,
          'outputTokens' => 5,
          'outputTokensDetails' => { 'reasoningTokens' => 7 }
        }
      }

      response = instance_double(Faraday::Response, body: response_body)
      message = described_class.parse_completion_response(response)

      expect(message.thinking_tokens).to eq(7)
    end

    it 'preserves every reasoning block from a multi-block turn, not just the first' do
      # Anthropic requires every thinking/redacted_thinking block from a turn to be replayed
      # unmodified. Interleaved/adaptive thinking can put more than one reasoningContent block
      # in a single turn (e.g. a redacted block followed by a normal one) — collapsing them to
      # a single merged text+signature silently drops data and the next request gets rejected
      # with "Invalid data in redacted_thinking block".
      response_body = {
        'output' => {
          'message' => {
            'content' => [
              { 'reasoningContent' => { 'redactedContent' => 'opaque-blob-1' } },
              { 'reasoningContent' => { 'reasoningText' => { 'text' => 'step two', 'signature' => 'sig-2' } } },
              { 'toolUse' => { 'toolUseId' => 't1', 'name' => 'search', 'input' => {} } }
            ]
          }
        },
        'usage' => {}
      }

      response = instance_double(Faraday::Response, body: response_body)
      message = described_class.parse_completion_response(response)

      expect(message.thinking.blocks).to eq(response_body['output']['message']['content'].first(2))
    end
  end

  describe '.format_tool_result_content' do
    it 'uses a placeholder when the tool returns no content' do
      result = described_class.format_tool_result_content('')

      expect(result).to eq([{ text: '(no output)' }])
    end
  end

  describe '.render_payload' do
    let(:model) do
      instance_double(RubyLLM::Model::Info,
                      id: 'anthropic.claude-haiku-4-5-20251001-v1:0',
                      max_tokens: nil,
                      metadata: {})
    end

    let(:base_args) do
      {
        tools: {},
        temperature: nil,
        model: model,
        stream: false
      }
    end

    def render_payload(messages = [], **overrides)
      described_class.render_payload(messages, **base_args, **overrides)
    end

    context 'when schema is provided' do
      let(:schema) do
        {
          name: 'response',
          schema: {
            type: 'object',
            properties: { name: { type: 'string' } },
            required: ['name'],
            additionalProperties: false
          },
          strict: true
        }
      end

      it 'includes outputConfig with stringified schema' do
        payload = render_payload(schema: schema)

        output_config = payload[:outputConfig]
        expect(output_config).not_to be_nil
        expect(output_config[:textFormat][:type]).to eq('json_schema')

        json_schema = output_config[:textFormat][:structure][:jsonSchema]
        expect(json_schema[:name]).to eq('response')
        expect(json_schema[:schema]).to be_a(String)

        parsed = JSON.parse(json_schema[:schema])
        expect(parsed['type']).to eq('object')
        expect(parsed['properties']).to eq({ 'name' => { 'type' => 'string' } })
      end

      it 'strips :strict from the schema' do
        payload = render_payload(schema: schema)

        json_schema = payload[:outputConfig][:textFormat][:structure][:jsonSchema]
        parsed = JSON.parse(json_schema[:schema])
        expect(parsed).not_to have_key('strict')
        expect(parsed).not_to have_key(:strict)
      end

      it 'uses schema name and inner schema' do
        custom_schema = RubyLLM::Utils.deep_dup(schema)
        custom_schema[:name] = 'PersonSchema'

        payload = render_payload(schema: custom_schema)

        json_schema = payload[:outputConfig][:textFormat][:structure][:jsonSchema]
        expect(json_schema[:name]).to eq('PersonSchema')

        parsed = JSON.parse(json_schema[:schema])
        expect(parsed['type']).to eq('object')
        expect(parsed['properties']).to eq({ 'name' => { 'type' => 'string' } })
        expect(parsed).not_to have_key('name')
        expect(parsed).not_to have_key('schema')
      end

      it 'does not mutate the original schema' do
        original = RubyLLM::Utils.deep_dup(schema)
        render_payload(schema: schema)
        expect(schema).to eq(original)
      end
    end

    context 'when schema is nil' do
      it 'does not include outputConfig' do
        payload = render_payload(schema: nil)
        expect(payload).not_to have_key(:outputConfig)
      end
    end

    it 'does not send finish_reason back to the provider' do
      message = RubyLLM::Message.new(role: :assistant, content: 'Done', finish_reason: 'MAX_TOKENS')

      payload = render_payload([message], schema: nil)

      expect(payload[:messages].first).not_to have_key(:finishReason)
      expect(payload[:messages].first[:content]).to eq([{ text: 'Done' }])
    end

    it 'replays every reasoning block from a multi-block turn unmodified, in order' do
      original_blocks = [
        { 'reasoningContent' => { 'redactedContent' => 'opaque-blob-1' } },
        { 'reasoningContent' => { 'reasoningText' => { 'text' => 'step two', 'signature' => 'sig-2' } } }
      ]
      thinking = RubyLLM::Thinking.build(text: 'step two', signature: 'sig-2', blocks: original_blocks)
      message = RubyLLM::Message.new(role: :assistant, content: 'Done', thinking: thinking)

      payload = render_payload([message], schema: nil)

      reasoning_blocks = payload[:messages].first[:content].select { |block| block['reasoningContent'] }
      expect(reasoning_blocks).to eq(original_blocks)
    end

    it 'reconstructs a single reasoning block when thinking has no raw blocks' do
      thinking = RubyLLM::Thinking.build(text: 'thought', signature: 'sig')
      message = RubyLLM::Message.new(role: :assistant, content: 'Done', thinking: thinking)

      payload = render_payload([message], schema: nil)

      reasoning_blocks = payload[:messages].first[:content].select { |block| block[:reasoningContent] }
      expect(reasoning_blocks).to eq([{ reasoningContent: { reasoningText: { text: 'thought', signature: 'sig' } } }])
    end

    it 'reconstructs a signature-only thinking turn as an empty reasoningText, never redactedContent' do
      # A persisted signature is always a real reasoningText signature (redacted blobs are
      # only ever captured into thinking.blocks). Wrapping it in redactedContent makes
      # Anthropic reject the next request with "Invalid `data` in `redacted_thinking` block".
      thinking = RubyLLM::Thinking.build(text: nil, signature: 'sig-only')
      message = RubyLLM::Message.new(role: :assistant, content: 'Done', thinking: thinking)

      payload = render_payload([message], schema: nil)

      reasoning_blocks = payload[:messages].first[:content].select { |block| block[:reasoningContent] }
      expect(reasoning_blocks).to eq([{ reasoningContent: { reasoningText: { text: '', signature: 'sig-only' } } }])
    end

    context 'when consecutive messages resolve to the same role' do
      # Bedrock Converse hard-rejects a payload where two messages in a row share a role
      # ("A conversation must alternate between user and assistant roles"). This happens when
      # a new user message is injected mid-tool-loop (e.g. from the AppFolio agents_app fork)
      # immediately after a tool result, since tool results are synthesized as role: 'user'.
      it 'merges multiple consecutive user messages into one' do
        messages = [
          RubyLLM::Message.new(role: :user, content: 'first'),
          RubyLLM::Message.new(role: :user, content: 'second')
        ]

        payload = render_payload(messages, schema: nil)

        expect(payload[:messages].size).to eq(1)
        expect(payload[:messages].first[:role]).to eq('user')
        expect(payload[:messages].first[:content]).to eq([{ text: 'first' }, { text: 'second' }])
      end

      it 'merges a user message that immediately follows a tool-result-flushed user message' do
        messages = [
          RubyLLM::Message.new(role: :assistant, content: 'thinking', tool_calls: {
                                 't1' => RubyLLM::ToolCall.new(id: 't1', name: 'search', arguments: {})
                               }),
          RubyLLM::Message.new(role: :user, content: 'result', tool_call_id: 't1'),
          RubyLLM::Message.new(role: :user, content: 'injected mid-loop message')
        ]

        payload = render_payload(messages, schema: nil)

        expect(payload[:messages].size).to eq(2)
        merged = payload[:messages].last
        expect(merged[:role]).to eq('user')
        expect(merged[:content]).to eq([
                                         { toolResult: { toolUseId: 't1', content: [{ text: 'result' }] } },
                                         { text: 'injected mid-loop message' }
                                       ])
      end

      it 'merges consecutive assistant messages into one' do
        messages = [
          RubyLLM::Message.new(role: :assistant, content: 'first'),
          RubyLLM::Message.new(role: :assistant, content: 'second')
        ]

        payload = render_payload(messages, schema: nil)

        expect(payload[:messages].size).to eq(1)
        expect(payload[:messages].first[:role]).to eq('assistant')
        expect(payload[:messages].first[:content]).to eq([{ text: 'first' }, { text: 'second' }])
      end

      it 'keeps toolResult blocks ahead of other content blocks regardless of merge order' do
        messages = [
          RubyLLM::Message.new(role: :assistant, content: 'thinking', tool_calls: {
                                 't1' => RubyLLM::ToolCall.new(id: 't1', name: 'search', arguments: {})
                               }),
          RubyLLM::Message.new(role: :user, content: 'injected before result', tool_call_id: nil),
          RubyLLM::Message.new(role: :user, content: 'result', tool_call_id: 't1')
        ]

        payload = render_payload(messages, schema: nil)

        expect(payload[:messages].size).to eq(2)
        merged = payload[:messages].last
        expect(merged[:role]).to eq('user')
        expect(merged[:content]).to eq([
                                         { toolResult: { toolUseId: 't1', content: [{ text: 'result' }] } },
                                         { text: 'injected before result' }
                                       ])
      end

      it 'hoists reasoningContent blocks ahead of toolResult and other blocks when merging assistant messages' do
        thinking = RubyLLM::Thinking.build(text: 'second thought', signature: 'sig-2')
        messages = [
          RubyLLM::Message.new(role: :assistant, content: 'first thought'),
          RubyLLM::Message.new(role: :assistant, content: 'second thought', thinking: thinking)
        ]

        payload = render_payload(messages, schema: nil)

        expect(payload[:messages].size).to eq(1)
        merged = payload[:messages].first
        expect(merged[:role]).to eq('assistant')
        expect(merged[:content]).to eq([
                                         { reasoningContent: { reasoningText: { text: 'second thought',
                                                                                signature: 'sig-2' } } },
                                         { text: 'first thought' },
                                         { text: 'second thought' }
                                       ])
      end
    end
  end
end
