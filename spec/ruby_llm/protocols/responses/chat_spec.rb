# frozen_string_literal: true

require 'spec_helper'

RSpec.describe RubyLLM::Protocols::Responses::Chat do
  let(:protocol) { RubyLLM::Protocols::Responses.allocate }
  let(:model) { instance_double(RubyLLM::Model::Info, id: 'gpt-5-nano') }

  def render_payload(messages, tools: {}, schema: nil, thinking: nil, stream: false)
    protocol.send(:render_payload, messages, tools:, temperature: nil, model:, stream:, schema:, thinking:,
                                             tool_prefs: nil)
  end

  describe '#reasoning_model?' do
    it 'matches bare reasoning model ids' do
      expect(protocol.send(:reasoning_model?, 'gpt-5-nano')).to be(true)
      expect(protocol.send(:reasoning_model?, 'o3')).to be(true)
    end

    it 'matches reasoning model ids with the openai. vendor prefix' do
      expect(protocol.send(:reasoning_model?, 'openai.gpt-5.6-sol')).to be(true)
      expect(protocol.send(:reasoning_model?, 'openai.o3')).to be(true)
    end

    it 'does not match non-reasoning ids' do
      expect(protocol.send(:reasoning_model?, 'gpt-4o')).to be(false)
      expect(protocol.send(:reasoning_model?, 'openai.gpt-oss-120b')).to be(false)
    end
  end

  describe '#render_payload' do
    it 'runs stateless and replays encrypted reasoning' do
      payload = render_payload([RubyLLM::Message.new(role: :user, content: 'hi')])

      expect(payload[:store]).to be(false)
      expect(payload[:include]).to eq(['reasoning.encrypted_content'])
    end

    it 'turns system messages into instructions' do
      messages = [
        RubyLLM::Message.new(role: :system, content: 'Be brief.'),
        RubyLLM::Message.new(role: :user, content: 'hi')
      ]

      payload = render_payload(messages)

      expect(payload[:instructions]).to eq('Be brief.')
      expect(payload[:input]).to eq([{ role: 'user', content: 'hi' }])
    end

    it 'replays reasoning, tool calls, and tool outputs as items' do
      messages = [
        RubyLLM::Message.new(role: :user, content: 'weather?'),
        RubyLLM::Message.new(
          role: :assistant,
          content: '',
          thinking: RubyLLM::Thinking.new(signature: 'ENCRYPTED'),
          tool_calls: { 'call_1' => RubyLLM::ToolCall.new(id: 'call_1', name: 'weather', arguments: {}) }
        ),
        RubyLLM::Message.new(role: :tool, content: 'Sunny', tool_call_id: 'call_1')
      ]

      payload = render_payload(messages)

      expect(payload[:input][1]).to eq({ type: 'reasoning', summary: [], encrypted_content: 'ENCRYPTED' })
      expect(payload[:input][2]).to eq({ type: 'function_call', call_id: 'call_1', name: 'weather',
                                         arguments: '{}' })
      expect(payload[:input][3]).to eq({ type: 'function_call_output', call_id: 'call_1', output: 'Sunny' })
    end

    it 'replays assistant text as output_text content' do
      messages = [
        RubyLLM::Message.new(role: :user, content: 'hi'),
        RubyLLM::Message.new(role: :assistant, content: 'Hello!', finish_reason: 'MAX_TOKENS')
      ]

      payload = render_payload(messages)

      expect(payload[:input][1]).to eq({ role: 'assistant', content: [{ type: 'output_text', text: 'Hello!' }] })
    end

    it 'uses flat function definitions' do
      tool = instance_double(RubyLLM::Tool, name: 'weather', description: 'Looks up weather',
                                            params_schema: { 'type' => 'object' }, provider_params: {})

      payload = render_payload([RubyLLM::Message.new(role: :user, content: 'hi')], tools: { weather: tool })

      expect(payload[:tools]).to eq([{
                                      type: 'function',
                                      name: 'weather',
                                      description: 'Looks up weather',
                                      parameters: { 'type' => 'object' }
                                    }])
    end

    it 'renders structured output as a text format' do
      schema = { name: 'response', schema: { type: 'object' }, strict: true }

      payload = render_payload([RubyLLM::Message.new(role: :user, content: 'hi')], schema: schema)

      expect(payload[:text]).to eq({
                                     format: {
                                       type: 'json_schema',
                                       name: 'response',
                                       schema: { type: 'object' },
                                       strict: true
                                     }
                                   })
    end

    context 'with a GPT-6 model' do
      let(:model) { instance_double(RubyLLM::Model::Info, id: 'openai.gpt-6-sol') }

      it 'replays every reasoning block as received' do
        blocks = [
          { 'type' => 'reasoning', 'id' => 'rs_1', 'summary' => [], 'encrypted_content' => 'ENC1' },
          { 'type' => 'reasoning', 'id' => 'rs_2', 'summary' => [], 'encrypted_content' => 'ENC2' }
        ]
        messages = [
          RubyLLM::Message.new(role: :user, content: 'hi'),
          RubyLLM::Message.new(role: :assistant, content: 'Hello!',
                               thinking: RubyLLM::Thinking.build(text: nil, signature: 'ENC1', blocks: blocks))
        ]

        payload = render_payload(messages)

        expect(payload[:input][1..]).to eq([*blocks,
                                            { role: 'assistant',
                                              content: [{ type: 'output_text', text: 'Hello!' }] }])
      end

      it 'ignores thinking blocks that are not Responses reasoning items' do
        messages = [
          RubyLLM::Message.new(role: :user, content: 'hi'),
          RubyLLM::Message.new(role: :assistant, content: 'Hello!',
                               thinking: RubyLLM::Thinking.build(text: nil, signature: 'ENC',
                                                                 blocks: [{ type: 'thinking', thinking: 'x' }]))
        ]

        payload = render_payload(messages)

        expect(payload[:input][1]).to eq({ type: 'reasoning', summary: [], encrypted_content: 'ENC' })
      end

      it 'replays raw output items verbatim, whether OutputItems or a stored Content::Raw' do
        items = [
          { 'type' => 'message', 'id' => 'msg_1', 'phase' => 'commentary',
            'content' => [{ 'type' => 'output_text', 'text' => 'Checking.' }] },
          { 'type' => 'function_call', 'id' => 'fc_1', 'call_id' => 'c1', 'name' => 'weather', 'arguments' => '{}' }
        ]
        tool_calls = { 'c1' => RubyLLM::ToolCall.new(id: 'c1', name: 'weather', arguments: {}) }

        [RubyLLM::Protocols::Responses::OutputItems.new(items), RubyLLM::Content::Raw.new(items)].each do |content|
          messages = [
            RubyLLM::Message.new(role: :user, content: 'weather?'),
            RubyLLM::Message.new(role: :assistant, content:, tool_calls:,
                                 thinking: RubyLLM::Thinking.build(text: nil, signature: 'ENC')),
            RubyLLM::Message.new(role: :tool, content: 'Sunny', tool_call_id: 'c1')
          ]

          payload = render_payload(messages)

          expect(payload[:input]).to eq([{ role: 'user', content: 'weather?' }, *items,
                                         { type: 'function_call_output', call_id: 'c1', output: 'Sunny' }])
        end
      end

      it 'replaces the input so far with the output of a compaction' do
        compaction = { 'object' => 'response.compaction',
                       'output' => [{ 'type' => 'compaction', 'encrypted_content' => 'CMP' }] }
        messages = [
          RubyLLM::Message.new(role: :system, content: 'Be brief.'),
          RubyLLM::Message.new(role: :user, content: 'old question'),
          RubyLLM::Message.new(role: :assistant, content: 'old answer'),
          RubyLLM::Message.new(role: :assistant, content: RubyLLM::Protocols::Responses::OutputItems.new(compaction)),
          RubyLLM::Message.new(role: :user, content: 'new question')
        ]

        payload = render_payload(messages)

        expect(payload[:instructions]).to eq('Be brief.')
        expect(payload[:input]).to eq([{ 'type' => 'compaction', 'encrypted_content' => 'CMP' },
                                       { role: 'user', content: 'new question' }])
      end

      it 'drops the input before a reply whose items hold a compaction' do
        items = [
          { 'type' => 'reasoning', 'encrypted_content' => 'ENC' },
          { 'type' => 'compaction', 'encrypted_content' => 'CMP' },
          { 'type' => 'message', 'content' => [{ 'type' => 'output_text', 'text' => '21' }] }
        ]
        messages = [
          RubyLLM::Message.new(role: :user, content: 'long history'),
          RubyLLM::Message.new(role: :assistant, content: RubyLLM::Protocols::Responses::OutputItems.new(items)),
          RubyLLM::Message.new(role: :user, content: 'follow-up')
        ]

        payload = render_payload(messages)

        expect(payload[:input]).to eq([*items.drop(1), { role: 'user', content: 'follow-up' }])
      end
    end

    it 'maps thinking effort to reasoning' do
      thinking = RubyLLM::Thinking::Config.new(effort: 'low')

      payload = render_payload([RubyLLM::Message.new(role: :user, content: 'hi')], thinking: thinking)

      expect(payload[:reasoning]).to eq({ effort: 'low' })
    end
  end

  describe '#parse_completion_response' do
    let(:response_model_id) { 'gpt-5-nano' }

    def response_with(output, usage: {})
      instance_double(
        Faraday::Response,
        body: { 'model' => response_model_id, 'output' => output, 'usage' => usage, 'status' => 'completed' }
      )
    end

    it 'joins output_text parts into content' do
      response = response_with([
                                 { 'type' => 'message',
                                   'content' => [{ 'type' => 'output_text', 'text' => 'Hello' },
                                                 { 'type' => 'output_text', 'text' => ' world' }] }
                               ])

      message = protocol.send(:parse_completion_response, response)

      expect(message.content).to eq('Hello world')
      expect(message.model_id).to eq('gpt-5-nano')
    end

    it 'parses function calls keyed by call_id' do
      response = response_with([
                                 { 'type' => 'function_call', 'call_id' => 'call_1', 'name' => 'weather',
                                   'arguments' => '{"city":"Berlin"}' }
                               ])

      message = protocol.send(:parse_completion_response, response)

      expect(message.tool_calls.keys).to eq(['call_1'])
      expect(message.tool_calls['call_1'].name).to eq('weather')
      expect(message.tool_calls['call_1'].arguments).to eq({ 'city' => 'Berlin' })
    end

    it 'parses reasoning summaries and encrypted content into thinking' do
      response = response_with([
                                 { 'type' => 'reasoning',
                                   'summary' => [{ 'type' => 'summary_text', 'text' => 'Thinking...' }],
                                   'encrypted_content' => 'ENCRYPTED' }
                               ])

      message = protocol.send(:parse_completion_response, response)

      expect(message.thinking.text).to eq('Thinking...')
      expect(message.thinking.signature).to eq('ENCRYPTED')
    end

    it 'maps usage with cached and reasoning tokens' do
      response = response_with([], usage: {
                                 'input_tokens' => 10,
                                 'output_tokens' => 7,
                                 'input_tokens_details' => { 'cached_tokens' => 4 },
                                 'output_tokens_details' => { 'reasoning_tokens' => 3 }
                               })

      message = protocol.send(:parse_completion_response, response)

      expect(message.input_tokens).to eq(6)
      expect(message.output_tokens).to eq(7)
      expect(message.cached_tokens).to eq(4)
      expect(message.thinking_tokens).to eq(3)
    end

    context 'with a GPT-6 model' do
      let(:response_model_id) { 'openai.gpt-6-sol' }

      def usage_for(details)
        response = response_with([], usage: { 'input_tokens' => 1000, 'output_tokens' => 5,
                                              'input_tokens_details' => details })
        message = protocol.send(:parse_completion_response, response)
        [message.input_tokens, message.cached_tokens, message.cache_creation_tokens]
      end

      it 'cache accounting: subtracts cache writes from input when the request only writes the cache' do
        expect(usage_for({ 'cached_tokens' => 0, 'cache_write_tokens' => 900 })).to eq([100, 0, 900])
      end

      it 'cache accounting: subtracts cache reads from input when the request only reads the cache' do
        expect(usage_for({ 'cached_tokens' => 900, 'cache_write_tokens' => 0 })).to eq([100, 900, 0])
      end

      it 'cache accounting: subtracts both when the request reads and writes the cache' do
        expect(usage_for({ 'cached_tokens' => 600, 'cache_write_tokens' => 300 })).to eq([100, 600, 300])
      end

      it 'cache accounting: leaves input whole when the response reports neither' do
        expect(usage_for({})).to eq([1000, nil, nil])
      end

      it 'keeps every reasoning item as a thinking block, with the first signature' do
        first = { 'type' => 'reasoning', 'id' => 'rs_1', 'summary' => [], 'encrypted_content' => 'ENC1' }
        second = { 'type' => 'reasoning', 'id' => 'rs_2', 'summary' => [], 'encrypted_content' => 'ENC2' }
        response = response_with([first, { 'type' => 'function_call', 'call_id' => 'c1', 'name' => 'a',
                                           'arguments' => '{}' }, second])

        message = protocol.send(:parse_completion_response, response)

        expect(message.thinking.signature).to eq('ENC1')
        expect(message.thinking.blocks).to eq([first, second])
        expect(message.content).to eq('')
      end

      it 'leaves a plain final answer as a String' do
        response = response_with([
                                   { 'type' => 'reasoning', 'summary' => [], 'encrypted_content' => 'ENC' },
                                   { 'type' => 'message', 'phase' => 'final_answer',
                                     'content' => [{ 'type' => 'output_text', 'text' => 'Done.' }] }
                                 ])

        message = protocol.send(:parse_completion_response, response)

        expect(message.content).to eq('Done.')
      end

      it 'keeps a reply with commentary as OutputItems while still parsing tool calls and thinking' do
        output = [
          { 'type' => 'reasoning', 'id' => 'rs_1', 'summary' => [], 'encrypted_content' => 'ENC' },
          { 'type' => 'message', 'id' => 'msg_1', 'status' => 'completed', 'phase' => 'commentary',
            'content' => [{ 'type' => 'output_text', 'text' => 'Checking.', 'annotations' => [] }] },
          { 'type' => 'function_call', 'id' => 'fc_1', 'call_id' => 'c1', 'name' => 'weather',
            'arguments' => '{}', 'status' => 'completed' }
        ]

        message = protocol.send(:parse_completion_response, response_with(output))

        expect(message.content).to be_a(RubyLLM::Protocols::Responses::OutputItems)
        expect(message.content.value).to eq(output)
        expect(message.content.text).to eq('')
        expect(message.content.commentary_text).to eq('Checking.')
        expect(message.tool_calls.keys).to eq(['c1'])
        expect(message.thinking.signature).to eq('ENC')
      end

      it 'keeps a reply with an unmodelled item type as OutputItems' do
        output = [
          { 'type' => 'compaction', 'encrypted_content' => 'CMP' },
          { 'type' => 'message', 'content' => [{ 'type' => 'output_text', 'text' => 'Hi' }] }
        ]

        message = protocol.send(:parse_completion_response, response_with(output))

        expect(message.content).to be_a(RubyLLM::Protocols::Responses::OutputItems)
        expect(message.content.text).to eq('Hi')
        expect(message.content.to_s).to eq('Hi')
      end
    end

    # Shape of a GPT-6 Sol tool-loop reply (reasoning, commentary, reasoning, function call), labelled as
    # GPT-5.6 Sol: models that exist today must parse it exactly as before.
    context 'with GPT-5.6 Sol' do
      let(:response_model_id) { 'openai.gpt-5.6-sol' }
      let(:output) do
        [
          { 'type' => 'reasoning', 'id' => 'rs_1', 'summary' => [], 'encrypted_content' => 'ENC1' },
          { 'type' => 'message', 'id' => 'msg_1', 'status' => 'completed', 'phase' => 'commentary',
            'role' => 'assistant',
            'content' => [{ 'type' => 'output_text', 'text' => 'Looking up Berlin.', 'annotations' => [] }] },
          { 'type' => 'reasoning', 'id' => 'rs_2', 'summary' => [], 'encrypted_content' => 'ENC2' },
          { 'type' => 'function_call', 'id' => 'fc_1', 'call_id' => 'c1', 'name' => 'city_code',
            'arguments' => '{"city":"Berlin"}', 'status' => 'completed' }
        ]
      end

      it 'keeps String content, the first signature and no blocks' do
        message = protocol.send(:parse_completion_response, response_with(output))

        expect(message.content).to eq('Looking up Berlin.')
        expect(message.thinking.signature).to eq('ENC1')
        expect(message.thinking.blocks).to be_nil
        expect(message.tool_calls.keys).to eq(['c1'])
      end

      it 'keeps the existing usage mapping, without cache-write accounting' do
        usage = { 'input_tokens' => 1000, 'output_tokens' => 5,
                  'input_tokens_details' => { 'cached_tokens' => 600, 'cache_write_tokens' => 300 } }

        message = protocol.send(:parse_completion_response, response_with(output, usage:))

        expect(message.input_tokens).to eq(400)
        expect(message.cached_tokens).to eq(600)
        expect(message.cache_creation_tokens).to be_nil
      end

      it 'replays only the first reasoning item and plain text, as before' do
        protocol.instance_variable_set(:@model, instance_double(RubyLLM::Model::Info, id: 'openai.gpt-5.6-sol'))
        reply = protocol.send(:parse_completion_response, response_with(output))
        sol = instance_double(RubyLLM::Model::Info, id: 'openai.gpt-5.6-sol')

        payload = protocol.send(:render_payload, [RubyLLM::Message.new(role: :user, content: 'weather?'), reply],
                                tools: {}, temperature: nil, model: sol, tool_prefs: nil)

        expect(payload[:input][1..]).to eq([
                                             { type: 'reasoning', summary: [], encrypted_content: 'ENC1' },
                                             { role: 'assistant',
                                               content: [{ type: 'output_text', text: 'Looking up Berlin.' }] },
                                             { type: 'function_call', call_id: 'c1', name: 'city_code',
                                               arguments: '{"city":"Berlin"}' }
                                           ])
      end
    end

    it 'does not synthesize finish_reason for completed function calls' do
      response = response_with([
                                 { 'type' => 'function_call', 'call_id' => 'call_1', 'name' => 'weather',
                                   'arguments' => '{}' }
                               ])

      message = protocol.send(:parse_completion_response, response)

      expect(message.finish_reason).to be_nil
    end

    it 'preserves incomplete_details reason as finish_reason when present' do
      response = instance_double(
        Faraday::Response,
        body: {
          'model' => 'gpt-5-nano',
          'output' => [],
          'status' => 'incomplete',
          'incomplete_details' => { 'reason' => 'max_output_tokens' }
        }
      )

      message = protocol.send(:parse_completion_response, response)

      expect(message.finish_reason).to eq('max_output_tokens')
    end
  end

  describe '#parse_compaction_response' do
    let(:model) { instance_double(RubyLLM::Model::Info, id: 'openai.gpt-6-sol') }

    before { protocol.instance_variable_set(:@model, model) }

    it 'returns the compaction as OutputItems with usage' do
      body = { 'object' => 'response.compaction', 'output' => [{ 'type' => 'compaction' }],
               'usage' => { 'input_tokens' => 50, 'output_tokens' => 10 } }

      message = protocol.send(:parse_compaction_response, instance_double(Faraday::Response, body:))

      expect(message.content).to be_a(RubyLLM::Protocols::Responses::OutputItems)
      expect(message.content.items).to eq([{ 'type' => 'compaction' }])
      expect(message.finish_reason).to eq('stop')
      expect(message.model_id).to eq('openai.gpt-6-sol')
      expect(message.input_tokens).to eq(50)
    end

    it 'raises when the body is not a response.compaction' do
      response = instance_double(Faraday::Response, body: { 'object' => 'response' })

      expect { protocol.send(:parse_compaction_response, response) }
        .to raise_error(RubyLLM::Error, /invalid compaction response/)
    end
  end

  describe '#compact' do
    it 'raises UnsupportedFeatureError for a model that is not GPT-6' do
      protocol.instance_variable_set(:@model, instance_double(RubyLLM::Model::Info, id: 'openai.gpt-5.6-sol'))

      expect { protocol.compact([RubyLLM::Message.new(role: :user, content: 'hi')]) }
        .to raise_error(RubyLLM::UnsupportedFeatureError, /only supported for GPT-6/)
    end
  end
end
