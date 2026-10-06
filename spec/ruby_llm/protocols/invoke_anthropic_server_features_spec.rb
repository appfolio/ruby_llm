# frozen_string_literal: true

require 'spec_helper'

# Live examples (VCR) of Anthropic's server-side context features through Bedrock InvokeModel,
# recorded in us-west-2. Each example is a multi-request conversation that sends the previous
# reply back unchanged, so the cassettes also hold the exact request and event formats the
# InvokeAnthropic protocol parses. Field names follow the Anthropic docs; where Bedrock
# rejects or ignores one, the example asserts what Bedrock does.
RSpec.describe RubyLLM::Protocols::InvokeAnthropic do
  include_context 'with configured RubyLLM'

  def invoke_chat(model_id)
    RubyLLM.chat(model: model_id, provider: :bedrock).with_protocol(:invoke_anthropic)
  end

  def user(text)
    RubyLLM::Message.new(role: :user, content: text)
  end

  # One completion for `messages`, on a fresh chat, without running any tool it asks for.
  def reply(model_id, messages, params: {}, tools: [], thinking: nil, &)
    chat = invoke_chat(model_id).with_tools(*tools).with_params(**params)
    chat.with_thinking(effort: thinking) if thinking
    chat.messages = messages
    chat.generate(&)
  end

  def build_tool(tool_name, tool_description, &result)
    Class.new(RubyLLM::Tool) do
      description tool_description
      param :key, desc: 'What to look up'
      define_method(:name) { tool_name }
      define_method(:execute) { |key:| result.call(key) }
    end.new
  end

  def ledger_lines(range)
    cities = %w[Lisbon Osaka Denver Nairobi Quito Tallinn Perth Hanoi Bergen Tucson Accra Cusco]
    goods = %w[copper-wire ceramic-tiles olive-oil bicycle-frames solar-panels wool-blankets glass-jars cedar-planks]
    range.map do |i|
      "Ledger entry #{i}: the depot in #{cities[i % cities.size]} received #{((i * 37) % 900) + 10} crates of " \
        "#{goods[i % goods.size]} on day #{((i * 13) % 365) + 1}."
    end.join("\n")
  end

  def blocks_of(message, type)
    message.content.value.select { |block| block['type'] == type }
  end

  def context_beta
    'context-management-2025-06-27'
  end

  def clear_tool_uses(trigger:, keep:)
    { type: 'clear_tool_uses_20250919', trigger: { type: 'input_tokens', value: trigger },
      keep: { type: 'tool_uses', value: keep } }
  end

  def compact(value)
    { context_management: { edits: [{ type: 'compact_20260112', trigger: { type: 'input_tokens', value: value } }] } }
  end

  def applied_edits(message)
    message.provider_data.dig('context_management', 'applied_edits')
  end

  # --- Example 1 -------------------------------------------------------------------------------

  def ledger_tool
    build_tool('read_ledger_page', 'Read one page of the shipping ledger by page number') do |key|
      page = key.to_i
      ledger_lines((((page - 1) * 40) + 1)..(page * 40))
    end
  end

  # Six earlier read_ledger_page round trips (about 1,500 tokens each), then a question.
  def ledger_history
    messages = [user('Read ledger pages 1 to 6 with read_ledger_page, one page per call.')]
    (1..6).each do |page|
      id = "toolu_ledger_#{page}"
      call = RubyLLM::ToolCall.new(id: id, name: 'read_ledger_page', arguments: { 'key' => page.to_s })
      messages << RubyLLM::Message.new(role: :assistant, content: nil, tool_calls: { id => call })
      messages << RubyLLM::Message.new(role: :tool, content: ledger_tool.execute(key: page.to_s), tool_call_id: id)
    end
    messages << user('Without calling any tool, name in one sentence the city of the last entry on page 6.')
  end

  def train_question(departure:)
    "A train leaves at #{departure} and travels 283 km at 71 km/h, stops 13 minutes, then travels 158 km " \
      'at 94 km/h. Reason carefully, then give only the arrival time to the minute.'
  end

  # --- Example 4 -------------------------------------------------------------------------------

  def catalog_tools
    verbs = %w[get list create update delete archive export import audit sync]
    nouns = %w[invoices customers orders refunds coupons vendors payroll timesheets badges parcels
               leases tenants budgets forecasts tickets surveys webinars licenses backups certificates]
    tools = verbs.product(nouns).map do |verb, noun|
      build_tool("#{verb}_#{noun}", "#{verb.capitalize} #{noun} records in the back-office system") { 'n/a' }
    end
    tools[0] = build_tool('lookup_tide_times', 'Look up high and low tide times for a harbor') do
      'High tide 6:12 AM, low tide 12:40 PM'
    end
    tools
  end

  def tide_question
    user('What are the tide times at Santa Barbara harbor today? Use the tool that provides them.')
  end

  # --- Examples 3 and 4, shared by every model they run on ------------------------------------

  def expect_compaction(model_id)
    params = { anthropic_beta: ['compact-2026-01-12'], max_tokens: 4000 }
    history = [user("#{ledger_lines(1..1600)}\n\nIn one sentence: what did the last ledger entry record?")]

    expect { reply(model_id, history, params: params.merge(compact(49_999))) }
      .to raise_error(RubyLLM::BadRequestError, /trigger.value must be at least 50000/)

    compacted = reply(model_id, history, params: params.merge(compact(50_000))) { |_chunk| nil }
    uncompacted = reply(model_id, history, params: params)

    expect(compacted.content).to be_a(described_class::ContentBlocks)
    expect(blocks_of(compacted, 'compaction')).to contain_exactly(include('content' => be_a(String)))
    expect(compacted.content.text).not_to be_empty
    expect_usage_excludes_compaction(compacted)
    expect(uncompacted.input_tokens).to be > 50_000

    history += [compacted, user('Which city was that depot in? One word.')]
    follow_up = reply(model_id, history, params: params.merge(compact(50_000)))

    expect(follow_up.content.to_s).not_to be_empty
    expect(follow_up.input_tokens).to be < 5000

    converse = RubyLLM.chat(model: model_id, provider: :bedrock).with_protocol(:converse)
    converse.messages = history
    expect { converse.generate }
      .to raise_error(RubyLLM::UnsupportedContentError, /cannot send compaction content blocks/)
  end

  # Top-level usage is the final `message` iteration only; the compaction step is billed on
  # top of it, so the cost of a compacted request is the sum of usage.iterations.
  def expect_usage_excludes_compaction(compacted)
    iterations = compacted.provider_data['iterations']
    expect(iterations.map { |iteration| iteration['type'] }).to eq(%w[compaction message])
    expect(iterations.first['input_tokens']).to be > 50_000
    expect([compacted.input_tokens, compacted.output_tokens])
      .to eq(iterations.last.values_at('input_tokens', 'output_tokens'))
    expect(iterations.sum { |iteration| iteration['input_tokens'] }).to be > compacted.input_tokens + 50_000
    expect(compacted.input_tokens).to be < 5000
  end

  def expect_tool_search(model_id)
    tools = catalog_tools
    search_tool = { type: 'tool_search_tool_regex_20251119', name: 'tool_search_tool_regex' }
    chat = invoke_chat(model_id).with_tools(*tools)
                                .with_params(max_tokens: 1500, deferred_tools: tools.map(&:name),
                                             server_tools: [search_tool])
    chat.messages = [tide_question]

    answer = chat.complete { |_chunk| nil }
    searched = chat.messages[1]

    expect(searched.content).to be_a(described_class::ContentBlocks)
    expect(blocks_of(searched, 'server_tool_use').first).to include(
      'name' => 'tool_search_tool_regex', 'id' => start_with('srvtoolu_'), 'input' => include('pattern')
    )
    references = blocks_of(searched, 'tool_search_tool_result').flat_map do |block|
      block.dig('content', 'tool_references').map { |reference| reference['tool_name'] }
    end
    expect(references).to include('lookup_tide_times')
    expect(searched.tool_calls.values.map(&:name)).to eq(['lookup_tide_times'])
    expect(chat.messages.map(&:role)).to eq(%i[user assistant tool assistant])
    expect(answer.content.to_s).to include('6:12')

    all_loaded = reply(model_id, [tide_question], params: { max_tokens: 1500 }, tools: tools)

    expect(searched.input_tokens).to be < all_loaded.input_tokens
  end

  # --- Examples 5 and 6 ------------------------------------------------------------------------

  # Each result carries about 1,500 tokens of movement history: Bedrock applies no
  # clear_tool_uses edit when the results it could clear are only a few tokens.
  def inventory_tool
    counts = { 'A-100' => 42, 'B-200' => 17 }
    build_tool('lookup_inventory', 'Look up the stock count for one SKU') do |key|
      "SKU #{key}: #{counts.fetch(key.strip.upcase, 0)} units\nMovement history:\n#{ledger_lines(1..40)}"
    end
  end

  def pallet_question
    'Each restock pallet holds 13 units. Look up SKU A-100, work out how many full pallets it fills and the ' \
      'remainder, then look up SKU B-200 and do the same, one call at a time. Reason carefully before each call. ' \
      'Finish with the combined pallets and remainder.'
  end

  def binding_params(behavior: 'drop_block', betas: [])
    thinking = behavior ? { thinking: { type: 'adaptive', block_binding: { prefix_mismatch_behavior: behavior } } } : {}
    { anthropic_beta: ['thinking-binding-controls-2026-08-01'] + betas, max_tokens: 8000 }.merge(thinking)
  end

  def without_thinking(message)
    RubyLLM::Message.new(role: :assistant, content: message.content, tool_calls: message.tool_calls,
                         model_id: message.model_id)
  end

  %w[us.anthropic.claude-opus-5-5 us.anthropic.claude-sonnet-5-5].each do |model_id|
    context "with #{model_id}" do
      it 'clear_tool_uses_20250919 clears old tool results and the next request succeeds' do
        history = ledger_history
        plain_params = { max_tokens: 300 }
        params = plain_params.merge(anthropic_beta: [context_beta],
                                    context_management: { edits: [clear_tool_uses(trigger: 3000, keep: 2)] })

        edited = reply(model_id, history, params: params, tools: [ledger_tool])
        unedited = reply(model_id, history, params: plain_params, tools: [ledger_tool])

        expect(applied_edits(edited)).to contain_exactly(
          hash_including('type' => 'clear_tool_uses_20250919', 'cleared_tool_uses' => 4,
                         'cleared_input_tokens' => be > 0)
        )
        expect(edited.input_tokens).to be < unedited.input_tokens
        expect(edited.tool_call?).to be(false)

        follow_up = reply(model_id, history + [edited, user('Which page numbers did you read? One line.')],
                          params: params, tools: [ledger_tool])

        expect(follow_up.content).to be_a(String).and(satisfy { |text| !text.empty? })
        expect(applied_edits(follow_up)).to contain_exactly(hash_including('cleared_tool_uses' => 4))
      end

      it 'clear_thinking_20251015 clears earlier thinking turns under adaptive thinking' do
        params = { anthropic_beta: [context_beta], max_tokens: 8000 }
        edits = { context_management: { edits: [{ type: 'clear_thinking_20251015',
                                                  keep: { type: 'thinking_turns', value: 1 } }] } }
        history = [user(train_question(departure: '9:47'))]
        first = reply(model_id, history, params: params, thinking: 'high')
        history += [first, user(train_question(departure: '11:23'))]
        second = reply(model_id, history, params: params, thinking: 'high')
        history += [second, user('How many minutes apart are your two arrival times? Only the number.')]

        cleared = reply(model_id, history, params: params.merge(edits), thinking: 'high')
        kept = reply(model_id, history, params: params, thinking: 'high')

        expect([first, second]).to all(satisfy { |message| message.thinking&.blocks&.any? })
        # Bedrock finding: keep `thinking_turns: 1` clears every earlier thinking turn (2 here), not all but one.
        expect(applied_edits(cleared)).to contain_exactly(
          hash_including('type' => 'clear_thinking_20251015', 'cleared_thinking_turns' => 2,
                         'cleared_input_tokens' => be > 0)
        )
        expect(cleared.input_tokens).to be < kept.input_tokens

        follow_up = reply(model_id, history + [cleared, user('Reply with OK.')], params: params.merge(edits),
                                                                                 thinking: 'high')

        expect(follow_up.content).to be_a(String).and(satisfy { |text| !text.empty? })
      end

      it 'compact_20260112 compacts at the 50000-token minimum and the compaction block replays' do
        expect_compaction(model_id)
      end

      it 'tool_search_tool_regex_20251119 finds a deferred tool and the replayed search blocks are accepted' do
        expect_tool_search(model_id)
      end

      it 'thinking-binding-controls-2026-08-01 drop_block reports dropped thinking for an edited replay only' do
        chat = invoke_chat(model_id).with_tool(inventory_tool).with_thinking(effort: 'high')
                                    .with_params(**binding_params)
        chat.ask(pallet_question)
        history = chat.messages.dup

        expect(history.count { |message| message.thinking&.blocks&.any? }).to be >= 2
        expect(history.last.provider_data['input_transformations']).to eq([])

        first_call = history.index { |message| message.role == :assistant }
        edited = history.each_with_index.map do |message, index|
          if index == first_call then without_thinking(message)
          elsif index == first_call + 1
            RubyLLM::Message.new(role: :tool, content: '[result removed]', tool_call_id: message.tool_call_id)
          else
            message
          end
        end
        edited << user('Repeat the total in one line.')

        dropped = reply(model_id, edited, params: binding_params, tools: [inventory_tool], thinking: 'high')

        expect(dropped.provider_data['input_transformations']).to all(
          include('type' => 'thinking_dropped', 'reason' => 'prefix_binding_mismatch',
                  'path' => start_with('messages.'))
        )
        expect(dropped.provider_data['input_transformations']).not_to be_empty

        unset = reply(model_id, edited, params: binding_params(behavior: nil), tools: [inventory_tool],
                                        thinking: 'high')

        expect(unset.provider_data['input_transformations']).to all(
          include('type' => 'thinking_mismatch_allowed', 'reason' => 'prefix_binding_mismatch')
        )
        expect(unset.provider_data['input_transformations'].size)
          .to eq(dropped.provider_data['input_transformations'].size)

        cleared = reply(model_id, history + [user('Repeat the total in one line.')],
                        params: binding_params(betas: [context_beta])
                                .merge(context_management: { edits: [clear_tool_uses(trigger: 1, keep: 1)] }),
                        tools: [inventory_tool], thinking: 'high')

        expect(applied_edits(cleared)).to contain_exactly(hash_including('type' => 'clear_tool_uses_20250919'))
        expect(cleared.provider_data['input_transformations']).to eq([])
      end

      it 'falls back to Converse mid tool loop and replays the InvokeModel thinking blocks' do
        # Thinking goes in additionalModelRequestFields, the format Converse callers already use;
        # InvokeModel lifts it to the top level. (Converse's own with_thinking(effort:) rendering
        # sends `reasoning_effort`, which Bedrock rejects for Sonnet 5.5.)
        thinking = { additionalModelRequestFields: { thinking: { type: 'adaptive' },
                                                     output_config: { effort: 'high' } } }
        chat = invoke_chat(model_id).with_tool(inventory_tool).with_params(max_tokens: 8000, **thinking)
        chat.ask_later(pallet_question)
        2.times { chat.step }
        call = chat.generate
        chat.run_tools

        expect(call.thinking.blocks).to include(include('type' => 'thinking', 'signature' => be_a(String)))
        expect(call.tool_call?).to be(true)

        chat.with_protocol(:converse).with_params(**thinking)
        replayed = chat.render[:messages][3][:content].first

        expect(replayed).to include(reasoningContent: include(reasoningText: include(signature: be_a(String))))

        answer = chat.generate

        expect(answer.content).to include('7')
      end

      it 'top-level cache_control caches the prefix automatically and the repeat request reads it' do
        history = [user("Shipping ledger for the cache check.\n#{ledger_lines(1..120)}\n\n" \
                        'How many crates did ledger entry 9 record? Only the number.')]
        params = { max_tokens: 300, cache_control: { type: 'ephemeral' } }

        written = reply(model_id, history, params: params)
        read = reply(model_id, history, params: params)

        expect(written.cache_creation_tokens).to be > 4000
        expect(written.cached_tokens).to eq(0)
        expect(written.provider_data['cache_creation'])
          .to eq('ephemeral_5m_input_tokens' => written.cache_creation_tokens, 'ephemeral_1h_input_tokens' => 0)
        expect(read.cached_tokens).to eq(written.cache_creation_tokens)
        expect(read.cache_creation_tokens).to eq(0)
        expect(read.input_tokens).to be < 50
      end

      it 'sends a system message placed after a user message as a system-role turn and Bedrock follows it' do
        chat = invoke_chat(model_id).with_instructions('You are terse.').with_params(max_tokens: 1000)
        chat.messages += [user('Name a fruit.'), RubyLLM::Message.new(role: :assistant, content: 'Apple.'),
                          user('Name another fruit.'),
                          RubyLLM::Message.new(role: :system, content: 'From now on, reply in uppercase only.')]

        payload = chat.render

        expect(payload[:system]).to eq([{ type: 'text', text: 'You are terse.' }])
        expect(payload[:messages].map { |message| message[:role] }).to eq(%w[user assistant user system])
        expect(payload[:messages].last)
          .to eq(role: 'system', content: [{ type: 'text', text: 'From now on, reply in uppercase only.' }])

        answer = chat.generate

        expect(answer.content).to match(/[A-Z]{3}/).and(eq(answer.content.upcase))
      end
    end
  end

  # Supernova's default chat model (Sonnet 5) and Opus 5: examples 3 and 4 only.
  %w[us.anthropic.claude-sonnet-5 us.anthropic.claude-opus-5].each do |model_id|
    context "with #{model_id}" do
      if model_id == 'us.anthropic.claude-sonnet-5'
        it 'compact_20260112 compacts at the 50000-token minimum and the compaction block replays' do
          expect_compaction(model_id)
        end
      end

      it 'tool_search_tool_regex_20251119 finds a deferred tool and the replayed search blocks are accepted' do
        expect_tool_search(model_id)
      end
    end
  end

  # Bedrock finding: Opus 5 accepts compact_20260112 (same 50000 minimum) but refuses the 58K-token
  # ledger prompt as `cyber`, with or without compaction, so no summary is produced.
  context 'with us.anthropic.claude-opus-5' do
    it 'compact_20260112 is accepted at the 50000-token minimum but the ledger prompt is refused' do
      model_id = 'us.anthropic.claude-opus-5'
      params = { anthropic_beta: ['compact-2026-01-12'], max_tokens: 4000 }
      history = [user("#{ledger_lines(1..1600)}\n\nIn one sentence: what did the last ledger entry record?")]

      expect { reply(model_id, history, params: params.merge(compact(49_999))) }
        .to raise_error(RubyLLM::BadRequestError, /trigger.value must be at least 50000/)

      compacted = reply(model_id, history, params: params.merge(compact(50_000))) { |_chunk| nil }
      uncompacted = reply(model_id, history, params: params)

      expect([compacted.finish_reason, uncompacted.finish_reason]).to eq(%w[refusal refusal])
      expect(compacted.content.value.map { |block| block['type'] }).to eq(['compaction'])
      expect(compacted.provider_data['iterations'].map { |iteration| iteration.values_at('type', 'output_tokens') })
        .to eq([['compaction', 0], ['message', 0]])
      expect(compacted.input_tokens).to be > 50_000
      expect(uncompacted.output_tokens).to eq(0)
    end
  end
end
