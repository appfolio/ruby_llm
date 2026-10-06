# frozen_string_literal: true

require 'spec_helper'

RSpec.describe RubyLLM::Protocols::Responses do
  include_context 'with configured RubyLLM'

  before { RubyLLM.config.bedrock_mantle_region = 'us-east-1' }

  gpt6_models = {
    'openai.gpt-6-luna' => { lowest_effort: 'none', sol: false, tool_loop: true },
    'openai.gpt-6-sol' => { lowest_effort: 'none', sol: true, tool_loop: true },
    'openai.gpt-6.1-sol' => { lowest_effort: 'low', sol: true, tool_loop: false }
  }

  let(:city_code_tool) do
    Class.new(RubyLLM::Tool) do
      def self.name = 'city_code'
      description 'Returns the internal code for a city. Forecasts can only be looked up by this code.'
      param :city, desc: 'City name, e.g. Berlin'

      def execute(city:)
        { 'berlin' => 'BER-7', 'paris' => 'PAR-3' }.fetch(city.to_s.downcase, 'UNK-0')
      end
    end
  end

  let(:forecast_tool) do
    Class.new(RubyLLM::Tool) do
      def self.name = 'forecast'
      description 'Returns the forecast for a city code obtained from the city_code tool.'
      param :code, desc: 'City code, e.g. BER-7'

      def execute(code:)
        { 'BER-7' => '18C, light rain', 'PAR-3' => '22C, sunny' }.fetch(code, 'unknown code')
      end
    end
  end

  def chat_for(model)
    RubyLLM.chat(model:, provider: :bedrock)
  end

  def facts_text(count)
    (1..count).map { |i| "Fact #{i}: warehouse #{i} stores #{i * 7} crates of item #{i * 13}." }.join("\n")
  end

  def output_of(message)
    Array(message.raw.body['output'])
  end

  def request_input(message)
    JSON.parse(message.raw.env.request_body)['input']
  end

  def reasoning_items(message)
    output_of(message).select { |item| item['type'] == 'reasoning' }
  end

  def expect_reasoning_replayed(assistant_messages)
    replayed = assistant_messages.each_cons(2).sum do |reply, next_reply|
      reasoning = reasoning_items(reply)
      expect(request_input(next_reply)).to include(*reasoning) unless reasoning.empty?
      reasoning.size
    end
    expect(replayed).to be_positive
  end

  def run_tool_loop(model)
    chat = chat_for(model).with_thinking(effort: 'high').with_tools(city_code_tool, forecast_tool, calls: :one)
    response = chat.ask('What is the forecast for Berlin, then for Paris? Look each city up one tool call ' \
                        'at a time. Before each tool call, tell me in a few words what you are about to ' \
                        'look up. Then answer in one short sentence per city.')
    [chat, response]
  end

  gpt6_models.each do |model, traits|
    describe model do
      it 'completes at the lowest and highest effort' do
        [traits[:lowest_effort], 'max'].each do |effort|
          response = chat_for(model).with_thinking(effort:).ask('Reply with exactly: OK')

          expect(response.content).to include('OK')
          expect(response.output_tokens).to be_positive
        end
      end

      # 6.1 Sol's tool-loop replies carried no reasoning item when recorded, so this example
      # runs only where a passing cassette exists.
      if traits[:tool_loop]
        it 'replays every reasoning item through a multi-round tool loop' do
          chat, response = run_tool_loop(model)
          replies = chat.messages.select { |msg| msg.role == :assistant }

          expect(replies.size).to be >= 3
          expect_reasoning_replayed(replies)
          expect(response.content.to_s).to match(/rain/i).and match(/sunny/i)

          if traits[:sol]
            commentary = replies.map(&:content).grep(RubyLLM::Protocols::Responses::OutputItems)
                                .find { |content| !content.commentary_text.empty? }
            expect(commentary).not_to be_nil
            expect(commentary.text).not_to include(commentary.commentary_text)
          end
        end
      end

      it 'compacts on the server when the input passes the threshold, and replays the compaction' do
        chat = chat_for(model).with_params(context_management: [{ type: 'compaction', compact_threshold: 1000 }])
        chat.add_message(role: :user, content: facts_text(120))
        reply = chat.ask('How many crates does warehouse 3 store? Answer with the number only.')

        expect(reply.input_tokens + reply.cached_tokens.to_i + reply.cache_creation_tokens.to_i).to be > 1500
        expect(reply.content).to be_a(RubyLLM::Protocols::Responses::OutputItems)
        expect(reply.content.items.map { |item| item['type'] }).to include('compaction')

        follow_up = chat.ask('What was your previous answer? Reply with the number only.')
        expect(follow_up.content.to_s).to include('21')
        expect(follow_up.input_tokens + follow_up.cached_tokens.to_i + follow_up.cache_creation_tokens.to_i)
          .to be < 500

        other = (gpt6_models.keys - [model]).first
        other_reply = chat.with_model(other, provider: :bedrock).ask('Repeat that number once more.')
        expect(other_reply.content.to_s).to include('21')
      end

      it 'streams a server compaction into the same content a sync reply holds' do
        chat = chat_for(model).with_params(context_management: [{ type: 'compaction', compact_threshold: 1000 }])
        chat.add_message(role: :user, content: facts_text(120))
        reply = chat.ask('How many crates does warehouse 3 store? Answer with the number only.') { |_chunk| nil }

        expect(reply.content).to be_a(RubyLLM::Protocols::Responses::OutputItems)
        expect(reply.content.items.map { |item| item['type'] }).to include('compaction')
        expect(reply.content.text).to include('21')

        follow_up = chat.ask('What was your previous answer? Reply with the number only.')
        expect(follow_up.content.to_s).to include('21')
      end

      it 'compacts a short conversation through /compact and continues from it' do
        chat = chat_for(model)
        chat.ask('My favourite colour is teal. Reply with exactly: NOTED')

        compaction = chat.compact_context

        expect(compaction.content.value['object']).to eq('response.compaction')
        expect(chat.messages.size).to eq(2)

        chat.add_message(compaction)
        follow_up = chat.ask('What is my favourite colour? Reply with one word.')
        expect(follow_up.content.to_s).to match(/teal/i)
      end

      it 'reports cache writes and cache reads separately from input' do
        prefix = facts_text(350)
        replies = %w[3 5].map do |warehouse|
          sleep 5 if VCR.current_cassette&.recording? && warehouse == '5'
          chat_for(model).with_params(prompt_cache_key: "ruby-llm-gpt6-cache-#{model}")
                         .ask("#{prefix}\n\nHow many crates does warehouse #{warehouse} store? Number only.")
        end

        expect(replies.first.cache_creation_tokens).to be_positive
        expect(replies.last.cached_tokens).to be_positive
        replies.each do |reply|
          usage = reply.raw.body['usage']
          expect(reply.input_tokens + reply.cached_tokens + reply.cache_creation_tokens).to eq(usage['input_tokens'])
          expect(reply.input_tokens).to be < 1000
        end
      end

      it 'rejects an over-length input and accepts a very high max_output_tokens' do
        expect { chat_for(model).with_thinking(effort: 'low').ask('word ' * 1_200_000) }
          .to raise_error(RubyLLM::ContextLengthExceededError)

        response = chat_for(model).with_thinking(effort: 'low').with_params(max_output_tokens: 5_000_000)
                                  .ask('Reply with exactly: OK')
        expect(response.content).to include('OK')
      end
    end
  end

  it 'rejects effort none on openai.gpt-6.1-sol' do
    expect { chat_for('openai.gpt-6.1-sol').with_thinking(effort: 'none').ask('Reply with exactly: OK') }
      .to raise_error(RubyLLM::BadRequestError, /'none' is not supported/)
  end
end
