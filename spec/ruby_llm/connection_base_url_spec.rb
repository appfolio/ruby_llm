# frozen_string_literal: true

require 'spec_helper'

RSpec.describe RubyLLM::Connection do
  let(:provider) do
    instance_double(
      RubyLLM::Provider,
      api_base: 'https://default.example.com',
      configured?: true,
      headers: {}
    )
  end

  let(:config) do
    instance_double(
      RubyLLM::Configuration,
      request_timeout: 300,
      max_retries: 3,
      retry_interval: 0.1,
      retry_interval_randomness: 0.5,
      retry_backoff_factor: 2,
      http_proxy: nil,
      log_regexp_timeout: 1.0,
      faraday_adapter: :net_http
    )
  end

  it 'defaults to provider.api_base' do
    connection = described_class.new(provider, config).connection

    expect(connection.url_prefix.to_s).to eq('https://default.example.com/')
  end

  it 'uses an explicit base_url when given, without calling provider.api_base' do
    connection = described_class.new(provider, config, base_url: 'https://override.example.com').connection

    expect(connection.url_prefix.to_s).to eq('https://override.example.com/')
    expect(provider).not_to have_received(:api_base)
  end
end
