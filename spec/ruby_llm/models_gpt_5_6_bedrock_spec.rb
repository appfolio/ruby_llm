# frozen_string_literal: true

require 'spec_helper'

RSpec.describe RubyLLM::Models do
  include_context 'with configured RubyLLM'

  # cache_write is verified, not a placeholder: per OpenAI's own docs
  # (https://developers.openai.com/api/docs/guides/prompt-caching), "Cache
  # writes have no additional fee on models before the GPT-5.6 family. For
  # GPT-5.6 models and later model families, cache writes cost 1.25x the
  # uncached input token rate." GPT-5.6 is the first OpenAI family to charge
  # for cache writes at all, which is why it's the only OpenAI family in this
  # registry carrying a cache_write_input_per_million value — every other
  # OpenAI entry has none because the rate really is $0 for those models.
  #
  # Luna/Terra base input/output rates reflect the July 30, 2026 OpenAI/Bedrock
  # price cut (AWS Bedrock pricing page: aws.amazon.com/bedrock/pricing; OpenAI's
  # July 30, 2026 pricing announcement). cache_read (0.1x) and cache_write (1.25x)
  # remain the same ratios of the (now lower) input rate as before. Sol was not
  # part of this price cut and is unchanged.
  #
  # Separately open (do NOT resolve here): OpenAI's developer community has
  # reported two GPT-5.6-specific usage-accounting bugs since launch (July
  # 2026) — one where cached_tokens + cache_write_tokens could nearly
  # double-count against prompt_tokens (OpenAI staff andyw1 confirmed this
  # and issued retroactive refunds), and a second, seemingly still-open one
  # alleging usage.output_tokens can be inflated ~9x by a reasoning-token
  # resummation bug, with the inflated figure being what's billed. We consume
  # these models via AWS Bedrock's mantle passthrough, not OpenAI's own API,
  # so it is unknown and unverified whether Bedrock's usage accounting
  # reproduces either bug or computes usage independently. Treat billed
  # Luna/Terra/Sol costs as an open risk until this is checked.
  {
    'openai.gpt-5.6-sol' => { input: 5.0, output: 30.0, cache_read: 0.5, cache_write: 6.25 },
    'openai.gpt-5.6-terra' => { input: 2.2, output: 13.2, cache_read: 0.22, cache_write: 2.75 },
    'openai.gpt-5.6-luna' => { input: 0.22, output: 1.32, cache_read: 0.022, cache_write: 0.275 }
  }.each do |id, cost|
    it "resolves #{id} from the bedrock provider with the documented effort values" do
      model = RubyLLM.models.find(id, :bedrock)

      expect(model.provider).to eq('bedrock')
      expect(model.reasoning_option_values('effort')).to eq(%w[none low medium high xhigh max])
      expect(model.metadata[:cost]).to include(cost)
    end
  end

  it 'resolves the bare-id aliases to their bedrock-qualified ids' do
    expect(RubyLLM.models.find('gpt-5.6-sol').id).to eq('openai.gpt-5.6-sol')
    expect(RubyLLM.models.find('gpt-5.6-terra').id).to eq('openai.gpt-5.6-terra')
    expect(RubyLLM.models.find('gpt-5.6-luna').id).to eq('openai.gpt-5.6-luna')
  end
end
