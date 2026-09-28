# frozen_string_literal: true

module RubyLLM
  module Protocols
    class InvokeAnthropic
      # Request rendering and response parsing for Bedrock InvokeModel with Anthropic Messages
      # bodies. Rendering reuses Anthropic::Chat's payload assembly (tools, tool_choice, system,
      # temperature, thinking, output_config) and overrides message formatting so it matches
      # Converse's replay guarantees: thinking blocks are always replayed verbatim, consecutive
      # same-role rows are folded, and tool results are batched into one user turn.
      module Chat
        ANTHROPIC_VERSION = 'bedrock-2023-05-31'
        DEFAULT_MAX_TOKENS = 4096

        THINKING_TYPES = %w[thinking redacted_thinking].freeze

        def completion_url
          "/model/#{escape_model_id(@model.id)}/invoke"
        end

        def escape_model_id(model_id)
          Converse::Chat.escape_model_id(model_id)
        end

        def render(messages, **)
          finalize_invoke_payload(super)
        end

        # The one place the InvokeModel envelope is assembled. `model`/`stream` never belong in
        # an InvokeModel body; anthropic_version must. Betas from params (`anthropic_beta:`)
        # and from invoke_anthropic_betas are merged into a single de-duplicated list, so a
        # later caller enabling a beta feature (e.g. context_management) adds the beta here
        # and the feature's own block via params, without touching the rest of rendering.
        def finalize_invoke_payload(payload)
          payload = payload.dup
          payload.delete(:model)
          payload.delete(:stream)
          betas = (Array(payload.delete(:anthropic_beta)) + invoke_anthropic_betas).map(&:to_s).uniq

          envelope = { anthropic_version: ANTHROPIC_VERSION }
          envelope[:anthropic_beta] = betas unless betas.empty?
          envelope.merge(payload)
        end

        def invoke_anthropic_betas
          []
        end

        def max_tokens_for(model)
          model.max_tokens || DEFAULT_MAX_TOKENS
        end

        # InvokeModel does not read the Anthropic Files API, so there is no provider file to
        # reference; large attachments are sent inline.
        def supports_provider_file_references?
          false
        end

        def build_system_content(system_messages)
          system_messages.flat_map do |msg|
            content = msg.content
            content.is_a?(RubyLLM::Content::Raw) ? Array(content.value) : format_content(content)
          end
        end

        def build_base_payload(chat_messages, model, _stream, thinking, citations: false)
          @render_citations = citations
          payload = {
            messages: format_messages(chat_messages),
            max_tokens: max_tokens_for(model)
          }

          add_thinking_fields(payload, thinking, model)
          payload
        end

        # Bedrock's model registry rarely carries Anthropic reasoning_options, so don't gate on
        # them the way the first-party protocol does; Bedrock validates the request itself.
        def build_thinking_payload(thinking, _model)
          return nil unless thinking&.enabled?

          effort = resolve_effort(thinking)
          return nil if effort == 'none'

          budget = resolve_budget(thinking)
          return enabled_thinking_payload(budget) if budget
          return adaptive_thinking_payload(effort) if effort

          raise ArgumentError, 'Anthropic adaptive thinking requires an effort'
        end

        def format_messages(messages)
          rendered = []
          tool_result_blocks = []

          messages.each do |msg|
            if msg.tool_result?
              tool_result_blocks.concat(format_tool_result_blocks(msg))
              next
            end

            unless tool_result_blocks.empty?
              append_message(rendered, { role: 'user', content: tool_result_blocks })
              tool_result_blocks = []
            end

            message = format_non_tool_message(msg)
            append_message(rendered, message) if message
          end

          append_message(rendered, { role: 'user', content: tool_result_blocks }) unless tool_result_blocks.empty?
          rendered
        end

        # Anthropic requires strict user/assistant alternation, so a synthesized tool-result
        # user turn immediately followed by a real user message is folded into one turn, as
        # Converse::Chat#append_message does.
        def append_message(rendered, message)
          previous = rendered.last

          if previous && previous[:role] == message[:role]
            previous[:content] = merge_content_blocks(previous[:content], message[:content])
          else
            rendered << message
          end
        end

        # Thinking blocks first, then tool_result blocks, then everything else, each group in
        # its original order: Anthropic requires tool_result blocks to lead a user turn and
        # thinking blocks to lead an assistant turn.
        def merge_content_blocks(existing_blocks, incoming_blocks)
          combined = existing_blocks + incoming_blocks
          thinking, rest = combined.partition { |block| thinking_block?(block) }
          tool_results, others = rest.partition { |block| block_type(block) == 'tool_result' }
          thinking + tool_results + others
        end

        def format_non_tool_message(msg)
          content = format_message_content(msg)
          return nil if content.empty?

          { role: format_role(msg.role), content: content }
        end

        def format_role(role)
          role == :assistant ? 'assistant' : 'user'
        end

        def format_message_content(msg)
          if msg.content.is_a?(RubyLLM::Content::Raw)
            blocks = Array(msg.content.value)
            return blocks if msg.role == :assistant

            return blocks.reject { |block| thinking_block?(block) }
          end

          blocks = []
          blocks.concat(format_thinking_blocks(msg.thinking) || []) if msg.role == :assistant
          blocks.concat(format_message_body(msg.content))
          blocks.concat(format_tool_use_blocks(msg)) if msg.tool_call?
          blocks
        end

        def format_message_body(content)
          return [] if content.nil? || (content.respond_to?(:empty?) && content.empty?)

          Array(format_content(content, citations: @render_citations)).reject do |block|
            block.is_a?(Hash) && block[:type] == 'text' && block[:text].to_s.empty?
          end
        end

        def format_tool_use_blocks(msg)
          msg.tool_calls.values.map do |tool_call|
            { type: 'tool_use', id: tool_call.id, name: tool_call.name, input: tool_call.arguments || {} }
          end
        end

        # Anthropic rejects a request whose latest assistant turn doesn't replay every
        # thinking/redacted_thinking block exactly as received, so blocks are replayed whenever
        # present (not only when thinking is enabled on this request, unlike the first-party
        # protocol). Blocks captured by the Converse protocol are translated to the Messages
        # shape so a chat can be served by either endpoint.
        def format_thinking_blocks(thinking)
          return nil unless thinking
          return thinking.blocks.filter_map { |block| anthropic_thinking_block(block) } if thinking.blocks
          return nil unless thinking.text || thinking.signature

          [{ type: 'thinking', thinking: thinking.text || '', signature: thinking.signature }.compact]
        end

        def anthropic_thinking_block(block)
          return nil unless block.is_a?(Hash)

          reasoning = indifferent_fetch(block, :reasoningContent)
          return block unless reasoning.is_a?(Hash)

          redacted = indifferent_fetch(reasoning, :redactedContent)
          return { type: 'redacted_thinking', data: redacted } if redacted

          text_block = indifferent_fetch(reasoning, :reasoningText) || {}
          {
            type: 'thinking',
            thinking: indifferent_fetch(text_block, :text) || '',
            signature: indifferent_fetch(text_block, :signature)
          }.compact
        end

        def indifferent_fetch(hash, key)
          hash[key.to_s] || hash[key]
        end

        def format_tool_result_blocks(msg)
          return Array(msg.content.value) if msg.content.is_a?(RubyLLM::Content::Raw) && raw_tool_result_turn?(msg)

          [format_tool_result_block(msg)]
        end

        # A Raw tool-result whose blocks are already complete tool_result blocks (the shape the
        # first-party protocol's format_tool_result passes through) is used as-is.
        def raw_tool_result_turn?(msg)
          blocks = Array(msg.content.value)
          blocks.any? && blocks.all? { |block| block_type(block) == 'tool_result' }
        end

        def format_tool_result_block(msg)
          {
            type: 'tool_result',
            tool_use_id: msg.tool_call_id,
            content: format_tool_result_content(msg.content)
          }
        end

        def format_tool_result_content(content)
          return format_raw_tool_result_content(content.value) if content.is_a?(RubyLLM::Content::Raw)
          return content.results.map { |r| Anthropic::Tools.search_result_block(r) } if search_results?(content)
          return [{ type: 'text', text: '(no output)' }] if blank_content?(content)

          blocks = format_message_body(content)
          blocks.empty? ? [{ type: 'text', text: '(no output)' }] : blocks
        end

        def format_raw_tool_result_content(raw_value)
          blocks = Array(raw_value).grep(Hash)
          blocks.empty? ? [{ type: 'text', text: raw_value.to_s }] : blocks
        end

        def search_results?(content)
          content.is_a?(RubyLLM::SearchResults)
        end

        def blank_content?(content)
          content.nil? || (content.respond_to?(:empty?) && content.empty?)
        end

        # Images and PDFs always travel base64-inline: InvokeModel has no URL fetch and no
        # Files API. Everything else matches Anthropic::Media.
        def format_content(content, citations: false)
          return Array(content.value) if content.is_a?(RubyLLM::Content::Raw)
          return [Anthropic::Media.format_text(content.to_json)] if content.is_a?(Hash) || content.is_a?(Array)
          return [Anthropic::Media.format_text(content)] unless content.is_a?(RubyLLM::Content)

          parts = []
          parts << Anthropic::Media.format_text(content.text) if content.text
          content.attachments.each { |attachment| parts << format_attachment(attachment, citations:) }
          parts
        end

        def format_attachment(attachment, citations:)
          raise UnsupportedAttachmentError, attachment.mime_type if attachment.provider_file?

          case attachment.type
          when :image then { type: 'image', source: base64_source(attachment) }
          when :pdf then format_inline_pdf(attachment, citations:)
          when :text
            citations ? Anthropic::Media.format_text_document(attachment) : Anthropic::Media.format_text_file(attachment)
          else
            raise UnsupportedAttachmentError, attachment.mime_type
          end
        end

        def format_inline_pdf(pdf, citations:)
          document = { type: 'document', source: base64_source(pdf) }
          Anthropic::Media.enable_citations(document, pdf) if citations
          document
        end

        def base64_source(attachment)
          { type: 'base64', media_type: attachment.mime_type, data: attachment.encoded }
        end

        # Same as Anthropic::Chat#build_message plus thinking.blocks: the raw thinking /
        # redacted_thinking blocks, which must be replayed verbatim on the next turn.
        def build_message(data, content, citations, thinking, thinking_signature, tool_use_blocks, raw) # rubocop:disable Metrics/ParameterLists
          usage = data['usage'] || {}
          blocks = raw_thinking_blocks(data['content'] || [])

          Message.new(
            role: :assistant,
            content: content,
            citations: citations,
            thinking: Thinking.build(text: thinking, signature: thinking_signature, blocks: blocks),
            tool_calls: Anthropic::Tools.parse_tool_calls(tool_use_blocks),
            input_tokens: usage['input_tokens'],
            output_tokens: usage['output_tokens'],
            cached_tokens: extract_cached_tokens(data),
            cache_creation_tokens: extract_cache_creation_tokens(data),
            thinking_tokens: usage.dig('output_tokens_details', 'thinking_tokens'),
            finish_reason: data['stop_reason'],
            model_id: data['model'],
            raw: raw
          )
        end

        def raw_thinking_blocks(content_blocks)
          blocks = content_blocks.select { |block| THINKING_TYPES.include?(block['type']) }
          blocks.empty? ? nil : blocks
        end

        def thinking_block?(block)
          return false unless block.is_a?(Hash)

          THINKING_TYPES.include?(block_type(block)) || block.key?(:reasoningContent) || block.key?('reasoningContent')
        end

        def block_type(block)
          return nil unless block.is_a?(Hash)

          (block[:type] || block['type'])&.to_s
        end
      end
    end
  end
end
