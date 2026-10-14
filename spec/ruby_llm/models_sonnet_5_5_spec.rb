# frozen_string_literal: true

require 'spec_helper'

RSpec.describe RubyLLM::Models do
  include_context 'with configured RubyLLM'

  # Prices (input/output/cache_read/cache_write per million tokens) come from
  # Anthropic's pricing page (platform.claude.com/docs/en/about-claude/pricing,
  # read 2026-10-14): global list price is 2 / 10 / 0.10 / 2.5, "Regional and
  # multi-region endpoints include a 10% premium over global endpoints", and
  # "Cache hits and refreshes on Claude Opus 5.5 and Claude Sonnet 5.5 are
  # priced at 0.05x the base input price." The AWS bill for 2026-10-05 to
  # 2026-10-12 (Cost Explorer, USW2-MP:USW2_* usage types) shows us. at
  # 2.20 / 11 / 0.11 / 2.75, which matches that rule.
  #
  # The global. and bare bedrock entries are at the global price with the
  # 0.05x cache read, and eu. is at the regional price. The au., jp.,
  # anthropic and vertexai entries are not called by Supernova and keep their
  # earlier values (cache read still at 0.1x input).
  #
  # Unlike Sonnet 5, these entries have no toggle reasoning option: Sonnet 5.5
  # rejects thinking: {type: "disabled"} (verified live on the
  # appfolio-ci-dev-tooling Bedrock account, 2026-10-05, outside this PR's
  # container) and only accepts adaptive thinking with output_config.effort,
  # the same as Opus 5.5.
  standard = { input: 2, output: 10, cache_read: 0.20, cache_write: 2.5 }
  global = { input: 2, output: 10, cache_read: 0.1, cache_write: 2.5 }
  regional = { input: 2.2, output: 11, cache_read: 0.11, cache_write: 2.75 }

  {
    %w[anthropic claude-sonnet-5-5] => standard,
    %w[vertexai claude-sonnet-5-5] => standard,
    %w[bedrock anthropic.claude-sonnet-5-5] => global,
    %w[bedrock global.anthropic.claude-sonnet-5-5] => global,
    %w[bedrock us.anthropic.claude-sonnet-5-5] => regional,
    %w[bedrock eu.anthropic.claude-sonnet-5-5] => regional,
    %w[bedrock au.anthropic.claude-sonnet-5-5] => standard,
    %w[bedrock jp.anthropic.claude-sonnet-5-5] => standard
  }.each do |(provider, id), cost|
    it "registers #{provider} #{id} with effort-only reasoning and the documented pricing" do
      # Match on provider and raw id: the anthropic and vertexai entries share
      # an id, and RubyLLM.models.find(id, :bedrock) renormalizes the region
      # prefix to the configured bedrock_region, which would mask the
      # EU-specific entry's own pricing.
      model = RubyLLM.models.all.find { |m| m.provider == provider && m.id == id }

      expect(model).not_to be_nil
      expect(model.reasoning_option_values('effort')).to eq(%w[low medium high xhigh max])
      expect(model.reasoning_option('toggle')).to be_nil
      expect(model.metadata[:cost]).to include(cost)
    end
  end

  it 'resolves the claude-sonnet-5-5 alias to the anthropic provider' do
    expect(RubyLLM.models.find('claude-sonnet-5-5').provider).to eq('anthropic')
  end

  it 'resolves us.anthropic.claude-sonnet-5-5 when bedrock_region is configured' do
    allow(RubyLLM.config).to receive(:bedrock_region).and_return('us-west-2')

    found = RubyLLM.models.find('us.anthropic.claude-sonnet-5-5', :bedrock)

    expect(found.id).to eq('us.anthropic.claude-sonnet-5-5')
    expect(found.provider).to eq('bedrock')
    expect(found.reasoning_option_values('effort')).to eq(%w[low medium high xhigh max])
  end
end
