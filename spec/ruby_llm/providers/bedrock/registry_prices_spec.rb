# frozen_string_literal: true

require 'spec_helper'

RSpec.describe 'Bedrock registry prices' do # rubocop:disable RSpec/DescribeClass
  def bedrock_model(id)
    RubyLLM.models.find(id, provider: 'bedrock')
  end

  def rates(tier)
    [tier.input_per_million, tier.output_per_million,
     tier.cache_read_input_per_million, tier.cache_write_input_per_million]
  end

  haiku_five_five_ids = %w[
    anthropic.claude-haiku-5-5
    au.anthropic.claude-haiku-5-5
    eu.anthropic.claude-haiku-5-5
    global.anthropic.claude-haiku-5-5
    jp.anthropic.claude-haiku-5-5
    us.anthropic.claude-haiku-5-5
  ]

  {
    'us.anthropic.claude-haiku-4-5-20251001-v1:0' => [1, 5, 0.1, 1.25],
    'us.anthropic.claude-opus-5-5' => [4.4, 22, 0.22, 5.5],
    'us.anthropic.claude-sonnet-5-5' => [2.2, 11, 0.11, 2.75],
    'us.anthropic.claude-haiku-5-5' => [0.11, 0.55, 0.011, 0.1375]
  }.each do |id, expected|
    it "prices #{id} at #{expected.join(' / ')}" do
      expect(rates(bedrock_model(id).pricing.text_tokens.standard)).to eq(expected)
    end
  end

  {
    'us.anthropic.claude-haiku-5-5' => [0.55, 2.75, 0.055, 0.6875],
    'global.anthropic.claude-haiku-5-5' => [0.5, 2.5, 0.05, 0.625]
  }.each do |id, expected|
    it "prices #{id} above 100,000 input tokens at #{expected.join(' / ')}" do
      text = bedrock_model(id).pricing.text_tokens

      expect(rates(text.long_context)).to eq(expected)
      expect(text.long_context_threshold).to eq(100_000)
      expect(rates(text.tier_for(100_001))).to eq(expected)
      expect(text.tier_for(100_000)).to equal(text.standard)
    end

    it "builds the same #{id} long-context tier from metadata.cost.tiers" do
      data = bedrock_model(id).to_h
      data[:pricing][:text_tokens] = data[:pricing][:text_tokens].slice(:standard)
      text = RubyLLM::Model.new(data).pricing.text_tokens

      expect(rates(text.long_context)).to eq(expected)
      expect(text.long_context_threshold).to eq(100_000)
    end
  end

  haiku_five_five_ids.each do |id|
    it "offers #{id} effort only, with no thinking toggle" do
      expect(bedrock_model(id).reasoning_options)
        .to eq([{ type: 'effort', values: %w[low medium high xhigh max] }])
    end
  end

  {
    'us.anthropic.claude-opus-5-5' => [1_000_000, 128_000],
    'us.anthropic.claude-sonnet-5-5' => [1_000_000, 128_000],
    'openai.gpt-6-sol' => [1_050_000, 128_000],
    'openai.gpt-6-luna' => [1_050_000, 128_000],
    'openai.gpt-6.1-sol' => [1_050_000, 128_000],
    'openai.gpt-6.1-terra' => [1_050_000, 128_000]
  }.each do |id, (context_window, max_output_tokens)|
    it "finds #{id} on bedrock with its limits" do
      model = bedrock_model(id)

      expect([model.provider, model.id]).to eq(['bedrock', id])
      expect([model.context_window, model.max_output_tokens]).to eq([context_window, max_output_tokens])
    end
  end

  %w[openai.gpt-6.1-terra us.openai.gpt-6.1-terra].each do |id|
    it "prices #{id} at the US Terra rates with a 272,000 tier" do
      text = bedrock_model(id).pricing.text_tokens

      expect(rates(text.standard)).to eq([1.1, 5.5, 0.055, 1.375])
      expect(rates(text.long_context)).to eq([2.2, 8.25, 0.11, 2.75])
      expect(text.long_context_threshold).to eq(272_000)
    end

    it "accepts effort none through max on #{id}" do
      expect(bedrock_model(id).reasoning_option_values(:effort)).to eq(%w[none low medium high xhigh max])
    end
  end

  it 'keeps openai.gpt-5.6-sol at its billed rates' do
    text = bedrock_model('openai.gpt-5.6-sol').pricing.text_tokens

    expect(rates(text.standard)).to eq([4.4, 22, 0.44, 5.5])
    expect(rates(text.long_context)).to eq([8.8, 33, 0.88, 11])
    expect(text.long_context_threshold).to eq(272_000)
  end
end
