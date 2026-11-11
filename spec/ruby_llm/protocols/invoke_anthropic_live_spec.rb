# frozen_string_literal: true

require 'spec_helper'

class InvokeAnthropicLiveWeather < RubyLLM::Tool
  description 'Get the current weather for a city'
  parameter :city, description: 'City name'

  def execute(city:)
    "It is 15 degrees Celsius and sunny in #{city}."
  end
end

RSpec.describe RubyLLM::Protocols::InvokeAnthropic, :live do
  def invoke_chat(model_id = 'us.anthropic.claude-opus-5-5')
    RubyLLM.chat(model: model_id, provider: :bedrock, protocol: :invoke_anthropic)
  end

  %w[us.anthropic.claude-opus-5-5 us.anthropic.claude-sonnet-5-5].each do |model_id|
    describe model_id do
      it 'answers through /invoke' do
        response = invoke_chat(model_id).ask('Say OK and nothing else.')

        expect(response.content).to include('OK')
        expect(response.raw.env.url.path).to end_with('/invoke')
        expect(response.tokens.input).to be_positive
        expect(response.provider_data).to be_a(Hash)
      end

      it 'streams through /invoke-with-response-stream' do
        chunks = []
        response = invoke_chat(model_id).ask('Say OK and nothing else.') { |chunk| chunks << chunk }

        expect(chunks.filter_map(&:content).join).to include('OK')
        expect(response.content).to include('OK')
        expect(response.raw.env.url.path).to end_with('/invoke-with-response-stream')
        expect(response.tokens.input).to be_positive
        expect(response.tokens.output).to be_positive
      end
    end
  end

  it 'accepts a prompt cache breakpoint for the whole request' do
    response = invoke_chat.with_caching.ask('Say OK and nothing else.')

    expect(response.content).to include('OK')
  end

  it 'finds a deferred tool through tool search and calls it, streamed' do
    weather = InvokeAnthropicLiveWeather.new.name
    chat = invoke_chat.with_tools(InvokeAnthropicLiveWeather).with_provider_options(
      deferred_tools: [weather],
      server_tools: [{ type: 'tool_search_tool_regex_20251119', name: 'tool_search_tool_regex' }]
    )

    response = chat.ask('Search your tools for a weather tool, then use it for Paris.') { |_chunk| nil }

    searched = chat.messages.select { |message| message.raw_content&.any? { |b| b['type'] == 'server_tool_use' } }
    called = chat.messages.select { |message| message.tool_calls&.values&.any? { |call| call.name == weather } }
    expect(searched).not_to be_empty
    expect(called).not_to be_empty
    expect(response.content).to include('15')
  end

  it 'compacts a padded history and replays the compaction on the next turn' do
    chat = invoke_chat.with_compaction(at: 50_000)
    padding = Array.new(2_400) { |i| "Line #{i}: the quick brown fox jumps over the lazy dog." }.join("\n")
    chat.add_message(role: :user, content: "Remember this log.\n#{padding}")
    chat.add_message(role: :assistant, content: 'Noted.')

    response = chat.ask('Summarize the log in one sentence.')
    follow_up = chat.ask('Say OK and nothing else.')

    expect(response.raw_content.map { |block| block['type'] }).to include('compaction')
    expect(follow_up.content).to include('OK')
  end

  it 'replays a reply with thinking on the next turn' do
    chat = invoke_chat.with_thinking(effort: 'low')

    first = chat.ask('What is 17 times 23? Answer with the number.')
    second = chat.ask('Add 1 to that and answer with the number.')

    expect(first.raw_reasoning).to include('anthropic')
    expect(first.content).to include('391')
    expect(second.content).to include('392')
  end
end
