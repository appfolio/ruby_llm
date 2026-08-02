# frozen_string_literal: true

require 'faraday'
require 'faraday/multipart'
require 'faraday/retry'
require 'ruby_llm/error_middleware'
require 'timeout'

module RubyLLM
  # Connection class for managing API connections to various providers.
  class Connection
    attr_reader :provider, :connection, :config

    def self.basic(&)
      Faraday.new do |f|
        f.response :logger,
                   RubyLLM.logger,
                   bodies: false,
                   errors: true,
                   headers: false,
                   log_level: :debug
        f.response :raise_error
        yield f if block_given?
      end
    end

    def initialize(provider, config, base_url: provider.api_base)
      @provider = provider
      @config = config

      @connection = Faraday.new(base_url) do |faraday|
        setup_timeout(faraday)
        setup_logging(faraday)
        setup_retry(faraday)
        setup_middleware(faraday)
        setup_http_proxy(faraday)
      end
    end

    def post(url, payload, &)
      instrument_request(:post, url) do
        @connection.post url, payload do |req|
          req.headers.merge! @provider.headers
          yield req if block_given?
        end
      end
    end

    def get(url, &)
      instrument_request(:get, url) do
        @connection.get url do |req|
          req.headers.merge! @provider.headers
          yield req if block_given?
        end
      end
    end

    # Keeps the config and Faraday internals out of pretty-printed output.
    def pretty_print_instance_variables
      super - %i[@config @connection]
    end

    private

    def instrument_request(method, url)
      payload = {
        provider: @provider.slug,
        method: method,
        url: url
      }

      RubyLLM.instrument('request.ruby_llm', payload, config: @config) do
        response = yield
        payload[:status] = response.status if response.respond_to?(:status)
        response
      end
    end

    def setup_timeout(faraday)
      faraday.options.timeout = @config.request_timeout
    end

    def setup_logging(faraday)
      faraday.response :logger,
                       RubyLLM.logger,
                       bodies: RubyLLM.logger.debug?,
                       errors: true,
                       headers: false,
                       log_level: :debug do |logger|
        logger.filter(logging_regexp('[A-Za-z0-9+/=]{100,}'), '[BASE64 DATA]')
        logger.filter(logging_regexp('[-\\d.e,\\s]{100,}'), '[EMBEDDINGS ARRAY]')
      end
    end

    def logging_regexp(pattern)
      return Regexp.new(pattern) if @config.log_regexp_timeout.nil? || !Regexp.respond_to?(:timeout)

      Regexp.new(pattern, timeout: @config.log_regexp_timeout)
    end

    def setup_retry(faraday)
      faraday.request :retry, {
        max: @config.max_retries,
        interval: @config.retry_interval,
        interval_randomness: @config.retry_interval_randomness,
        backoff_factor: @config.retry_backoff_factor,
        methods: Faraday::Retry::Middleware::IDEMPOTENT_METHODS + [:post],
        exceptions: retry_exceptions,
        retry_block: method(:resign_bedrock_retry)
      }
    end

    # SigV4 signatures embed the request timestamp and expire after 5 minutes. Bedrock
    # requests are signed once, before the first attempt, by the protocol code that builds
    # them — so without this, faraday-retry would resend that same stale signature on every
    # retry, guaranteeing "Signature expired" on any retry that follows a slow first attempt.
    # retry_block runs synchronously, still inside the retry middleware, right before the env
    # is replayed, so recomputing the signature here re-signs each attempt with a fresh
    # X-Amz-Date. Other providers don't define #sign_headers and are unaffected.
    def resign_bedrock_retry(env:, **)
      return unless @provider.respond_to?(:sign_headers)

      env.request_headers.merge!(
        @provider.sign_headers(env.method.to_s.upcase, env.url.request_uri, env.body.to_s, base_url: @provider.api_base)
      )
    end

    def setup_middleware(faraday)
      faraday.request :multipart
      faraday.request :json
      faraday.response :json
      faraday.adapter(@config.faraday_adapter)
      faraday.use :llm_errors, provider: @provider
    end

    def setup_http_proxy(faraday)
      return unless @config.http_proxy

      faraday.proxy = @config.http_proxy
    end

    def retry_exceptions
      [
        Errno::ETIMEDOUT,
        Timeout::Error,
        Faraday::TimeoutError,
        Faraday::ConnectionFailed,
        Faraday::RetriableResponse,
        RubyLLM::RateLimitError,
        RubyLLM::ServerError,
        RubyLLM::ServiceUnavailableError,
        RubyLLM::OverloadedError
      ]
    end
  end
end
