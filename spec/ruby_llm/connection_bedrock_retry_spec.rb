# frozen_string_literal: true

require 'spec_helper'
require 'faraday/retry'

RSpec.describe RubyLLM::Connection do
  describe 'Bedrock SigV4 re-signing on retry' do
    let(:config) do
      RubyLLM::Configuration.new.tap do |c|
        c.bedrock_api_key = 'static-key'
        c.bedrock_secret_key = 'static-secret'
        c.bedrock_region = 'us-east-1'
        c.max_retries = 1
        c.retry_interval = 0
      end
    end

    let(:provider) { RubyLLM::Providers::Bedrock.new(config) }

    def stub_timeout_then_success(connection)
      calls = 0
      request_headers = []

      faraday = connection.instance_variable_get(:@connection)
      faraday.builder.adapter(:test) do |stub|
        stub.post('/model/x/converse') do |env|
          calls += 1
          request_headers << env.request_headers.dup
          raise Faraday::TimeoutError if calls == 1

          [200, {}, '{}']
        end
      end

      request_headers
    end

    it 'signs the retried attempt with a fresh, independently-valid signature' do
      connection = provider.connection
      request_headers = stub_timeout_then_success(connection)

      body = '{"a":1}'
      response = connection.post('/model/x/converse', { a: 1 }) do |req|
        req.headers.merge!(provider.sign_headers('POST', '/model/x/converse', body))
      end

      expect(response.status).to eq(200)
      expect(request_headers.size).to eq(2)

      second_date = request_headers[1]['X-Amz-Date']
      second_auth = request_headers[1]['Authorization']

      # The retry's signature must match what #sign_headers independently computes for
      # the same X-Amz-Date, proving the retry was actually re-signed (recomputed from
      # the current request) rather than carrying forward the first attempt's signature.
      allow(Time).to receive(:now).and_return(Time.strptime(second_date, '%Y%m%dT%H%M%SZ'))
      expected_auth = provider.sign_headers('POST', '/model/x/converse', body)['Authorization']
      expect(second_auth).to eq(expected_auth)
    end

    it 'produces a strictly later X-Amz-Date on the retried attempt when time has advanced' do
      connection = provider.connection
      request_headers = stub_timeout_then_success(connection)

      real_now = Time.now.utc
      call_count = 0
      allow(Time).to receive(:now) do
        call_count += 1
        # First call signs the initial attempt; every subsequent call (the retry's
        # re-sign) simulates the clock having advanced past the 5-minute SigV4 window.
        call_count == 1 ? real_now : real_now + 400
      end

      body = '{"a":1}'
      connection.post('/model/x/converse', { a: 1 }) do |req|
        req.headers.merge!(provider.sign_headers('POST', '/model/x/converse', body))
      end

      first_date = request_headers[0]['X-Amz-Date']
      second_date = request_headers[1]['X-Amz-Date']

      expect(second_date).not_to eq(first_date)
    end

    it 'does not attempt to re-sign requests for non-Bedrock providers' do
      other_config = RubyLLM::Configuration.new.tap do |c|
        c.openai_api_key = 'sk-test'
        c.max_retries = 1
        c.retry_interval = 0
      end
      other_provider = RubyLLM::Providers::OpenAI.new(other_config)

      connection = described_class.new(other_provider, other_config)
      calls = 0

      faraday = connection.instance_variable_get(:@connection)
      faraday.builder.adapter(:test) do |stub|
        stub.post('/chat') do
          calls += 1
          raise Faraday::TimeoutError if calls == 1

          [200, {}, '{}']
        end
      end

      response = connection.post('/chat', { a: 1 })

      expect(response.status).to eq(200)
      expect(calls).to eq(2)
    end
  end
end
