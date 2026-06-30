# frozen_string_literal: true

module RubyLLM
  module Protocols
    # AWS Bedrock InvokeModel API – sends the raw Anthropic Messages wire format.
    # Reuses Converse's SigV4 signing and AWS event-stream decode framing;
    # overrides only URL paths, payload layout, and event parsing to speak
    # the native Anthropic protocol instead of Converse.
    class BedrockInvokeModel < Converse
      include BedrockInvokeModel::Chat
      include BedrockInvokeModel::Streaming
    end
  end
end
