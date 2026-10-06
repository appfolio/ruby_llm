# frozen_string_literal: true

require 'spec_helper'

RSpec.describe RubyLLM::Chat do
  include_context 'with configured RubyLLM'

  def basic_chat(model:, provider:)
    chat = RubyLLM.chat(model: model, provider: provider)
    return chat.with_params(enable_thinking: false) if provider == :gpustack && model == 'qwen3'

    chat
  end

  def total_input_tokens(message)
    message.input_tokens.to_i + message.cached_tokens.to_i + message.cache_creation_tokens.to_i
  end

  def expect_token_usage(message)
    expect(total_input_tokens(message)).to be_positive
    expect(message.output_tokens).to be_positive
  end

  describe 'basic chat functionality' do
    CHAT_MODELS.each do |model_info|
      model = model_info[:model]
      provider = model_info[:provider]
      it "#{provider}/#{model} can have a basic conversation" do
        chat = basic_chat(model: model, provider: provider)
        response = chat.ask("What's 2 + 2?")

        expect(response.content).to include('4')
        expect(response.role).to eq(:assistant)
        expect_token_usage(response)
      end

      it "#{provider}/#{model} returns raw responses" do
        chat = RubyLLM.chat(model: model, provider: provider)
        response = chat.ask('What is the capital of France?')
        expect(response.raw).to be_present
        expect(response.raw.headers).to be_present
        expect(response.raw.body).to be_present
        expect(response.raw.status).to be_present
        expect(response.raw.status).to eq(200)
        expect(response.raw.env.request_body).to be_present
      end

      it "#{provider}/#{model} can handle multi-turn conversations" do
        chat = basic_chat(model: model, provider: provider)

        first = chat.ask('Who is the creator of the programming language Ruby?')
        expect(first.content).to include('Matz')

        followup = chat.ask('What year did he create Ruby?')
        expect(followup.content).to include('199')
      end

      it "#{provider}/#{model} successfully uses the system prompt" do
        chat = RubyLLM.chat(model: model, provider: provider).with_temperature(0.0)

        # Use a distinctive and unusual instruction that wouldn't happen naturally
        chat.with_instructions 'You must include the exact phrase "XKCD7392" somewhere in your response.'

        response = chat.ask('Tell me about the weather.')
        expect(response.content).to match(/XKCD7392/i)
      end

      it "#{provider}/#{model} replaces previous system messages by default" do
        if %i[perplexity mistral].include?(provider)
          skip 'Provider API does not allow system messages after user/assistant messages'
        end
        skip 'xAI may retain prior instruction artifacts from conversation history' if provider == :xai

        if provider == :ollama && model == 'qwen3'
          skip 'ollama/qwen3 includes thinking tags even with enable_thinking: false'
        end

        chat = RubyLLM.chat(model: model, provider: provider).with_temperature(0.0)
        chat = chat.with_params(enable_thinking: false) if provider == :gpustack && model == 'qwen3'

        # Use a distinctive and unusual instruction that wouldn't happen naturally
        chat.with_instructions 'You must include the exact phrase "XKCD7392" somewhere in your response.'

        response = chat.ask('Tell me about the weather.')
        expect(response.content).to match(/XKCD7392/i)

        # Test ability to follow multiple instructions with another unique marker
        chat.with_instructions 'You must include the exact phrase "PURPLE-ELEPHANT-42" somewhere in your response.'

        response = chat.ask('What are some good books?')
        expect(response.content).not_to match(/XKCD7392/i)
        expect(response.content).to match(/PURPLE-ELEPHANT-42/i)
      end
    end
  end

  describe 'change model on the fly' do
    CHAT_MODELS.first(3).combination(2).each do |first, second|
      next if [first[:provider], second[:provider]] == %i[azure bedrock]

      it "between #{first[:provider]}/#{first[:model]} and #{second[:provider]}/#{second[:model]}" do
        chat = RubyLLM.chat(model: first[:model], provider: first[:provider]).with_temperature(0.0)
        response = chat.ask('Reply with exactly: FOUR')

        expect(response.content).to match(/four/i)
        expect(response.role).to eq(:assistant)
        expect_token_usage(response)

        chat.with_model(second[:model], provider: second[:provider])
        response = chat.ask('Reply with exactly: EIGHT')

        expect(response.content).to match(/eight/i)
        expect(response.role).to eq(:assistant)
        expect_token_usage(response)
      end
    end
  end

  describe '#compact_context' do
    it 'raises UnsupportedFeatureError when the protocol has no compaction endpoint' do
      chat = RubyLLM.chat(model: 'claude-3-5-haiku-20241022', provider: :anthropic)
      chat.add_message(role: :user, content: 'hi')

      expect do
        chat.compact_context
      end.to raise_error(RubyLLM::UnsupportedFeatureError, /no standalone compaction endpoint/)
      expect(chat.messages.size).to eq(1)
    end

    it 'leaves Enumerable#compact returning the messages' do
      chat = RubyLLM.chat(model: 'claude-3-5-haiku-20241022', provider: :anthropic)
      chat.add_message(role: :user, content: 'hi')

      expect(chat.to_a.compact).to eq(chat.messages)
    end
  end

  describe 'with_schema on an OutputItems reply' do
    it 'parses the final answer text as JSON and leaves commentary out' do
      items = [
        { 'type' => 'message', 'phase' => 'commentary',
          'content' => [{ 'type' => 'output_text', 'text' => 'Working it out.' }] },
        { 'type' => 'message', 'phase' => 'final_answer',
          'content' => [{ 'type' => 'output_text', 'text' => '{"city":"Berlin","code":"BER-7"}' }] }
      ]
      reply = RubyLLM::Message.new(role: :assistant, content: RubyLLM::Protocols::Responses::OutputItems.new(items))
      chat = RubyLLM.chat(model: 'gpt-6-sol', provider: :bedrock)
                    .with_schema({ type: 'object', properties: { city: { type: 'string' } } })
      allow(chat.instance_variable_get(:@provider)).to receive(:complete).and_return(reply)

      response = chat.ask('Which city?')

      expect(response.content).to eq({ 'city' => 'Berlin', 'code' => 'BER-7' })
    end
  end

  describe '#cost' do
    let(:model) do
      RubyLLM::Model::Info.new(
        id: 'priced-model',
        name: 'Priced Model',
        provider: 'openai',
        pricing: {
          text_tokens: {
            standard: {
              input_per_million: 1.0,
              output_per_million: 2.0
            }
          }
        }
      )
    end

    it 'sums message costs for the conversation' do
      allow(RubyLLM.models).to receive(:find).and_call_original
      allow(RubyLLM.models).to receive(:find).with('priced-model').and_return(model)

      chat = RubyLLM.chat(model: RubyLLM.config.default_model)
      chat.add_message(role: :user, content: 'Hello')
      chat.add_message(role: :assistant, content: 'Hi', input_tokens: 1_000, output_tokens: 2_000,
                       model_id: 'priced-model')
      chat.add_message(role: :assistant, content: 'Again', input_tokens: 500, output_tokens: 100,
                       model_id: 'priced-model')

      expect(chat.cost.input).to eq(0.0015)
      expect(chat.cost.output).to eq(0.0042)
      expect(chat.cost.total).to eq(0.0057)
    end

    it 'uses the chat model when a response model id cannot be resolved' do
      allow(RubyLLM.models).to receive(:find).and_call_original
      allow(RubyLLM.models).to receive(:find).with('priced-model', nil).and_return(model)
      allow(RubyLLM.models).to receive(:find).with('provider-backend-version').and_raise(RubyLLM::ModelNotFoundError)

      chat = RubyLLM.chat(model: 'priced-model')
      chat.add_message(role: :assistant, content: 'Hi', input_tokens: 1_000, output_tokens: 2_000,
                       model_id: 'provider-backend-version')

      expect(chat.cost.total).to eq(0.005)
    end
  end
end
