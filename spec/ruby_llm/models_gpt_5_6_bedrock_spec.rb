# frozen_string_literal: true

require 'spec_helper'

RSpec.describe RubyLLM::Models do
  include_context 'with configured RubyLLM'

  {
    'openai.gpt-5.6-sol' => { input: 5.0, output: 30.0, cache_read: 0.5 },
    'openai.gpt-5.6-terra' => { input: 2.5, output: 15.0, cache_read: 0.25 },
    'openai.gpt-5.6-luna' => { input: 1.0, output: 6.0, cache_read: 0.1 }
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
