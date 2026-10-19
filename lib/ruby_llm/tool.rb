# frozen_string_literal: true

require 'ruby_llm/schema'

module RubyLLM
  # Parameter definition for Tool methods. `desc:` is the 1.x name for `description:`.
  class Parameter
    attr_reader :name, :type, :description, :required

    def initialize(name, type: 'string', desc: nil, description: nil, required: true)
      @name = name
      @type = type
      @description = desc || description
      @required = required
    end
  end

  # Base class for creating tools that AI models can use
  class Tool
    POSITIONAL_PARAMETER_KINDS = %i[req opt rest].freeze

    class << self
      def description(text = nil)
        return @description unless text

        @description = text
      end
      alias desc description

      # RubyLLM 2.0 name. Stores into declared_parameters, as 2.0 does.
      def parameter(name, **options)
        declared_parameters[name] = Parameter.new(name, **options)
      end

      # 1.x name for parameter.
      def param(name, **options)
        parameter(name, **options)
      end

      # RubyLLM 2.0 name for the declared parameter hash (the storage).
      def declared_parameters
        @declared_parameters ||= {}
      end

      # Called with a schema or a block: the 2.0 form of params. Called with neither: the 1.x
      # getter for the parameter hash (2.0 raises ArgumentError here instead; this fork keeps
      # the getter until Supernova no longer calls it).
      def parameters(schema = nil, &block)
        return declared_parameters if schema.nil? && block.nil?

        define_parameters_schema(schema, block)
      end

      # 1.x name for parameters(schema) / parameters do ... end.
      def params(schema = nil, &block)
        define_parameters_schema(schema, block)
      end

      attr_reader :parameters_schema_definition

      # 1.x name for parameters_schema_definition.
      def params_schema_definition
        parameters_schema_definition
      end

      def with_params(**params)
        @provider_params = params
        self
      end

      def provider_params
        @provider_params ||= {}
      end

      private

      def define_parameters_schema(schema, block)
        @parameters_schema_definition = SchemaDefinition.new(schema:, block:)
        self
      end
    end

    def name
      klass_name = self.class.name
      normalized = klass_name.to_s.dup.force_encoding('UTF-8').unicode_normalize(:nfkd)
      ascii_name = normalized.encode('ASCII', replace: '').gsub(/[^a-zA-Z0-9_-]/, '-')
      Utils.underscore(ascii_name).delete_suffix('_tool')
    end

    def description
      self.class.description
    end

    # 1.x name for declared_parameters.
    def parameters
      declared_parameters
    end

    # RubyLLM 2.0 name. Calls the class through its 1.x getter so a class-level override of
    # either name is honored.
    def declared_parameters
      self.class.parameters
    end

    def provider_params
      self.class.provider_params
    end

    # 1.x name for parameters_schema.
    def params_schema
      parameters_schema
    end

    # RubyLLM 2.0 name; holds what params_schema did before.
    def parameters_schema
      return @parameters_schema if defined?(@parameters_schema)

      @parameters_schema = begin
        definition = self.class.params_schema_definition
        if definition&.present?
          definition.json_schema
        elsif parameters.any?
          SchemaDefinition.from_parameters(parameters)&.json_schema
        else
          SchemaDefinition.from_parameters(inferred_parameters, allow_empty: true)&.json_schema
        end
      end
    end

    def call(args)
      normalized_args = normalize_args(args)
      validation_error = validate_keyword_arguments(normalized_args)
      return { error: "Invalid tool arguments: #{validation_error}" } if validation_error

      RubyLLM.logger.debug { "Tool #{name} called with: #{normalized_args.inspect}" }
      result = execute(**normalized_args)
      RubyLLM.logger.debug { "Tool #{name} returned: #{result.inspect}" }
      result
    end

    def execute(...)
      raise NotImplementedError, 'Subclasses must implement #execute'
    end

    protected

    def normalize_args(args)
      return {} if args.nil?
      return args.transform_keys(&:to_sym) if args.respond_to?(:transform_keys)

      {}
    end

    def validate_keyword_arguments(arguments)
      required_keywords, optional_keywords, accepts_extra_keywords, accepts_positional_arguments =
        execute_keyword_signature

      return nil if required_keywords.empty? && optional_keywords.empty? && accepts_positional_arguments

      argument_keys = arguments.keys
      missing_keyword = (required_keywords - argument_keys).first
      return "missing keyword: #{missing_keyword}" if missing_keyword
      return nil if accepts_extra_keywords

      unknown_keyword = (argument_keys - (required_keywords + optional_keywords)).first
      return "unknown keyword: #{unknown_keyword}" if unknown_keyword

      nil
    end

    def execute_keyword_signature
      keyword_signature = method(:execute).parameters
      required_keywords = keyword_signature.filter_map { |kind, name| name if kind == :keyreq }
      optional_keywords = keyword_signature.filter_map { |kind, name| name if kind == :key }
      accepts_extra_keywords = keyword_signature.any? { |kind, _| kind == :keyrest }
      accepts_positional_arguments = keyword_signature.any? do |kind, _|
        POSITIONAL_PARAMETER_KINDS.include?(kind)
      end

      [required_keywords, optional_keywords, accepts_extra_keywords, accepts_positional_arguments]
    end

    def inferred_parameters
      required_keywords, optional_keywords, = execute_keyword_signature

      (required_keywords + optional_keywords).to_h do |name|
        [name, Parameter.new(name, required: required_keywords.include?(name))]
      end
    end

    # Wraps schema handling for tool parameters, supporting JSON Schema hashes,
    # RubyLLM::Schema instances/classes, and DSL blocks.
    class SchemaDefinition
      def self.from_parameters(parameters, allow_empty: false)
        return nil if parameters.nil? || (parameters.empty? && !allow_empty)

        properties = parameters.to_h do |name, param|
          schema = {
            type: map_type(param.type),
            description: param.description
          }.compact

          schema[:items] = default_items_schema if schema[:type] == 'array'

          [name.to_s, schema]
        end

        required = parameters.select { |_, param| param.required }.keys.map(&:to_s)

        json_schema = {
          type: 'object',
          properties: properties,
          required: required,
          additionalProperties: false,
          strict: true
        }

        new(schema: json_schema)
      end

      def self.map_type(type)
        case type.to_s
        when 'integer', 'int' then 'integer'
        when 'number', 'float', 'double' then 'number'
        when 'boolean' then 'boolean'
        when 'array' then 'array'
        when 'object' then 'object'
        else
          'string'
        end
      end

      def self.default_items_schema
        { type: 'string' }
      end

      def initialize(schema: nil, block: nil)
        @schema = schema
        @block = block
      end

      def present?
        @schema || @block
      end

      def json_schema
        @json_schema ||= RubyLLM::Utils.deep_stringify_keys(resolve_schema)
      end

      private

      def resolve_schema
        return resolve_direct_schema(@schema) if @schema
        return build_from_block(&@block) if @block

        nil
      end

      def resolve_direct_schema(schema)
        return extract_schema(schema.to_json_schema) if schema.respond_to?(:to_json_schema)
        return RubyLLM::Utils.deep_dup(schema) if schema.is_a?(Hash)
        if schema.is_a?(Class) && schema.method_defined?(:to_json_schema)
          return extract_schema(schema.new.to_json_schema)
        end

        nil
      end

      def build_from_block(&)
        schema_class = RubyLLM::Schema.create(&)
        extract_schema(schema_class.new.to_json_schema)
      end

      def extract_schema(schema_hash)
        return nil unless schema_hash.is_a?(Hash)

        schema = schema_hash[:schema] || schema_hash['schema'] || schema_hash
        RubyLLM::Utils.deep_dup(schema)
      end
    end
  end
end
