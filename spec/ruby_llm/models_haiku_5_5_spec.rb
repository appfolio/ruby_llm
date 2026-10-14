# frozen_string_literal: true

require 'spec_helper'

RSpec.describe RubyLLM::Models do
  include_context 'with configured RubyLLM'

  # Prices (input/output/cache_read/cache_write per million tokens, the cache
  # write at the 5-minute rate) come from the AWS price list (us-west-2,
  # AmazonBedrockFoundationModels) and Anthropic's pricing page, which agree
  # (both read 2026-10-14): global list price is 0.1 / 0.5 / 0.01 / 0.125, and
  # us. is at the regional rate, 10% over global: 0.11 / 0.55 / 0.011 / 0.1375.
  # The global., bare bedrock and anthropic entries are at the global price.
  #
  # The long-context tier starts above 100,000 prompt tokens, not 200,000 as
  # on other tiered models. Anthropic: "Claude Haiku 5.5 is priced by prompt
  # length: a request whose prompt is over 100,000 tokens pays higher prices.
  # A request's prompt length counts all of its input tokens, including cache
  # reads and cache writes." For that reason the entries have no legacy
  # context_over_200k key.
  #
  # These entries have no toggle reasoning option. Supernova never sends
  # thinking: {type: "disabled"} to this model, and Bedrock rejects disabled
  # thinking with effort xhigh or max (verified live on the
  # appfolio-ci-dev-tooling Bedrock account in us-west-2, 2026-10-14, outside
  # this PR's container).
  global = { input: 0.1, output: 0.5, cache_read: 0.01, cache_write: 0.125 }
  global_long = { input: 0.5, output: 2.5, cache_read: 0.05, cache_write: 0.625 }
  regional = { input: 0.11, output: 0.55, cache_read: 0.011, cache_write: 0.1375 }
  regional_long = { input: 0.55, output: 2.75, cache_read: 0.055, cache_write: 0.6875 }

  {
    %w[anthropic claude-haiku-5-5] => [global, global_long],
    %w[bedrock anthropic.claude-haiku-5-5] => [global, global_long],
    %w[bedrock global.anthropic.claude-haiku-5-5] => [global, global_long],
    %w[bedrock us.anthropic.claude-haiku-5-5] => [regional, regional_long]
  }.each do |(provider, id), (cost, long_context)|
    it "registers #{provider} #{id} with effort-only reasoning and the documented pricing" do
      # Match on provider and raw id: RubyLLM.models.find(id, :bedrock)
      # renormalizes the region prefix to the configured bedrock_region.
      model = RubyLLM.models.all.find { |m| m.provider == provider && m.id == id }

      expect(model).not_to be_nil
      expect(model.reasoning_option_values('effort')).to eq(%w[low medium high xhigh max])
      expect(model.reasoning_option('toggle')).to be_nil
      expect(model.supports?(:structured_output)).to be(true)
      expect(model.metadata[:cost]).to include(cost)

      tier = model.metadata[:cost][:tiers].first
      expect(tier[:tier]).to eq(type: 'context', size: 100_000)
      expect(tier).to include(long_context)
    end
  end

  it 'resolves the claude-haiku-5-5 alias to the anthropic provider' do
    expect(RubyLLM.models.find('claude-haiku-5-5').provider).to eq('anthropic')
  end

  it 'resolves us.anthropic.claude-haiku-5-5 when bedrock_region is configured' do
    allow(RubyLLM.config).to receive(:bedrock_region).and_return('us-west-2')

    found = RubyLLM.models.find('us.anthropic.claude-haiku-5-5', :bedrock)

    expect(found.id).to eq('us.anthropic.claude-haiku-5-5')
    expect(found.provider).to eq('bedrock')
    expect(found.reasoning_option_values('effort')).to eq(%w[low medium high xhigh max])
  end

  it 'resolves bare anthropic.claude-haiku-5-5 to the us. inference profile in us-west-2' do
    # The bare foundation model is callable through an inference profile only,
    # so the lookup must take the region prefix from the us. entry's
    # inference_types.
    allow(RubyLLM.config).to receive(:bedrock_region).and_return('us-west-2')

    found = RubyLLM.models.find('anthropic.claude-haiku-5-5', :bedrock)

    expect(found.id).to eq('us.anthropic.claude-haiku-5-5')
    expect(found.provider).to eq('bedrock')
  end
end
