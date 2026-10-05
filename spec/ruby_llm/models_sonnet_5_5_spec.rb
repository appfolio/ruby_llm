# frozen_string_literal: true

require 'spec_helper'

RSpec.describe RubyLLM::Models do
  include_context 'with configured RubyLLM'

  # Pricing is unchanged from each id's Claude Sonnet 5 entry: every id except
  # eu. is 2 / 10 / 0.20 / 2.5 (input/output/cache_read/cache_write per million
  # tokens), and eu. is 2.2 / 11 / 0.22 / 2.75. Source: the AWS Marketplace rate
  # card for the Claude Sonnet 5.5 offer accepted on the appfolio-ci-dev-tooling
  # account, global standard tier, checked 2026-10-05.
  #
  # Unlike Sonnet 5, these entries have no toggle reasoning option: Sonnet 5.5
  # rejects thinking: {type: "disabled"} (verified live on the
  # appfolio-ci-dev-tooling Bedrock account, 2026-10-05, outside this PR's
  # container) and only accepts adaptive thinking with output_config.effort,
  # the same as Opus 5.5.
  standard = { input: 2, output: 10, cache_read: 0.20, cache_write: 2.5 }
  eu = { input: 2.2, output: 11, cache_read: 0.22, cache_write: 2.75 }

  {
    %w[anthropic claude-sonnet-5-5] => standard,
    %w[vertexai claude-sonnet-5-5] => standard,
    %w[bedrock anthropic.claude-sonnet-5-5] => standard,
    %w[bedrock global.anthropic.claude-sonnet-5-5] => standard,
    %w[bedrock us.anthropic.claude-sonnet-5-5] => standard,
    %w[bedrock eu.anthropic.claude-sonnet-5-5] => eu,
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
