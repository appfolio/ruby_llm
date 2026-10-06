# frozen_string_literal: true

module RubyLLM
  module Protocols
    # The OpenAI Responses API. Overrides the chat surface of Chat Completions;
    # embeddings, images, moderation, and transcription are inherited. Runs
    # stateless (store: false) and replays encrypted reasoning so multi-turn
    # tool calls work without server-side state.
    class Responses < ChatCompletions
      include Responses::Chat
      include Responses::Media
      include Responses::Streaming
      include Responses::Tools

      # Compacts the conversation through the standalone /compact endpoint and returns the
      # response.compaction object as an assistant Message holding OutputItems. GPT-6 only.
      def compact(messages, headers: {})
        unless gpt6_model?(model.id)
          raise UnsupportedFeatureError, "Standalone compaction is only supported for GPT-6 models, not #{model.id}"
        end

        compaction_response(render_compaction_payload(messages), headers)
      end

      # Reasoning models reject the temperature parameter on this API.
      def maybe_normalize_temperature(temperature, model)
        return super unless reasoning_model?(model.id)

        unless temperature.nil?
          RubyLLM.logger.debug { "Model #{model.id} does not accept temperature on the Responses API, removing" }
        end
        nil
      end

      private

      def compaction_response(payload, additional_headers = {})
        response = @connection.post compaction_url, payload do |req|
          req.headers = additional_headers.merge(req.headers) unless additional_headers.empty?
        end
        parse_compaction_response response
      end
    end
  end
end
