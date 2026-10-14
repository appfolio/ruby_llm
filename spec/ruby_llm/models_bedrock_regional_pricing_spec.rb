# frozen_string_literal: true

require 'spec_helper'

RSpec.describe RubyLLM::Models do
  include_context 'with configured RubyLLM'

  # A us. inference profile is a regional Bedrock endpoint. Anthropic's pricing
  # page (platform.claude.com/docs/en/about-claude/pricing, read 2026-10-14),
  # under "Regional and multi-region endpoint pricing for Claude 4.5 models and
  # beyond", says: "Bedrock offers two endpoint types: global endpoints
  # (dynamic routing for maximum availability) and regional endpoints
  # (guaranteed data routing through specific geographic regions)." and
  # "Regional and multi-region endpoints include a 10% premium over global
  # endpoints."
  #
  # Cache reads use the same page's footnotes. Footnote 1: "Cache hits and
  # refreshes on Claude Fable 5.1 and Claude Mythos 5.1 are priced at 0.025x
  # the base input price." Footnote 2: "Cache hits and refreshes on Claude
  # Opus 5.5 and Claude Sonnet 5.5 are priced at 0.05x the base input price."
  # "All other models use the standard 0.1x multiplier."
  #
  # The AWS bill for 2026-10-05 to 2026-10-12 (Cost Explorer, cost divided by
  # usage, USW2-MP:USW2_* usage types) matches this rule: Sonnet 5 on us.
  # billed 2.20 / 11 / 0.22 / 2.75 and Sonnet 5.5 on us. billed
  # 2.20 / 11 / 0.11 / 2.75 (input/output/cache_read/cache_write per million
  # tokens).
  #
  # Unchanged on purpose:
  # - Haiku 4.5 billed 1.034 / 5.17 / 0.1034 / 1.2925, which is 94% of its
  #   regional rate, for a reason not yet known. Its global price
  #   (1 / 5 / 0.10 / 1.25) is closer to the bill, so it stays at that price.
  # - Opus 4.1 is older than Claude 4.5, so it has no regional premium.
  # - Sonnet 4 has no pricing in the registry.
  sonnet45 = { input: 3.3, output: 16.5, cache_read: 0.33, cache_write: 4.125 }
  opus45 = { input: 5.5, output: 27.5, cache_read: 0.55, cache_write: 6.875 }
  regional = {
    'us.anthropic.claude-sonnet-4-5-20250929-v1:0' => sonnet45,
    'us.anthropic.claude-sonnet-4-6' => sonnet45,
    'us.anthropic.claude-opus-4-5-20251101-v1:0' => opus45,
    'us.anthropic.claude-opus-4-6-v1' => opus45,
    'us.anthropic.claude-opus-4-7' => opus45,
    'us.anthropic.claude-opus-4-8' => opus45,
    'us.anthropic.claude-opus-5' => opus45,
    'us.anthropic.claude-sonnet-5' => { input: 2.2, output: 11, cache_read: 0.22, cache_write: 2.75 },
    'us.anthropic.claude-sonnet-5-5' => { input: 2.2, output: 11, cache_read: 0.11, cache_write: 2.75 },
    'us.anthropic.claude-opus-5-5' => { input: 4.4, output: 22, cache_read: 0.22, cache_write: 5.5 },
    'us.anthropic.claude-fable-5' => { input: 11, output: 55, cache_read: 1.1, cache_write: 13.75 },
    'us.anthropic.claude-fable-5-1' => { input: 11, output: 55, cache_read: 0.275, cache_write: 13.75 }
  }

  unchanged = {
    'us.anthropic.claude-haiku-4-5-20251001-v1:0' => { input: 1, output: 5, cache_read: 0.1, cache_write: 1.25 },
    'us.anthropic.claude-opus-4-1-20250805-v1:0' => { input: 15, output: 75, cache_read: 1.5, cache_write: 18.75 }
  }

  # Match on provider and raw id: RubyLLM.models.find(id, :bedrock)
  # renormalizes the region prefix to the configured bedrock_region.
  def bedrock_entry(id)
    RubyLLM.models.all.find { |m| m.provider == 'bedrock' && m.id == id }
  end

  def standard_pricing(model)
    model.pricing.text_tokens.standard.to_h.slice(
      :input_per_million, :output_per_million,
      :cache_read_input_per_million, :cache_write_input_per_million
    )
  end

  def expected_standard(cost)
    {
      input_per_million: cost[:input],
      output_per_million: cost[:output],
      cache_read_input_per_million: cost[:cache_read],
      cache_write_input_per_million: cost[:cache_write]
    }
  end

  regional.merge(unchanged).each do |id, cost|
    it "prices #{id} at #{cost.values.join(' / ')} in both price fields" do
      model = bedrock_entry(id)

      expect(model).not_to be_nil
      expect(model.metadata[:cost]).to eq(cost)
      expect(standard_pricing(model)).to eq(expected_standard(cost))
    end
  end

  it 'keeps us.anthropic.claude-sonnet-4-20250514-v1:0 without pricing' do
    model = bedrock_entry('us.anthropic.claude-sonnet-4-20250514-v1:0')

    expect(model).not_to be_nil
    expect(model.metadata[:cost]).to be_nil
    expect(model.pricing.to_h).to eq({})
  end
end
