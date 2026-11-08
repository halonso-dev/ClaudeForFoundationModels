// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

import Foundation

/// Request body for `POST /v1/messages`.
package struct MessagesRequest: Sendable, Codable {
  package var model: String
  package var maxTokens: Int
  package var system: String?
  package var messages: [Message]
  package var tools: [ToolDefinition]?
  package var toolChoice: ToolChoice?
  package var thinking: ThinkingConfig?
  package var temperature: Double?
  package var topP: Double?
  package var topK: Int?
  package var cacheControl: CacheControl?
  package var outputConfig: OutputConfig?
  /// Substitute models for a policy refusal. Sending any needs the
  /// ``Fallbacks/betaHeader`` opt-in.
  package var fallbacks: Fallbacks?
  package var stream: Bool

  package init(
    model: String,
    maxTokens: Int = 16_000,
    system: String? = nil,
    messages: [Message],
    tools: [ToolDefinition]? = nil,
    toolChoice: ToolChoice? = nil,
    thinking: ThinkingConfig? = nil,
    temperature: Double? = nil,
    topP: Double? = nil,
    topK: Int? = nil,
    cacheControl: CacheControl? = nil,
    outputConfig: OutputConfig? = nil,
    fallbacks: Fallbacks? = nil,
    stream: Bool = false
  ) {
    self.model = model
    self.maxTokens = maxTokens
    self.system = system
    self.messages = messages
    self.tools = tools
    self.toolChoice = toolChoice
    self.thinking = thinking
    self.temperature = temperature
    self.topP = topP
    self.topK = topK
    self.cacheControl = cacheControl
    self.outputConfig = outputConfig
    self.fallbacks = fallbacks
    self.stream = stream
  }

  private enum CodingKeys: String, CodingKey {
    case model, system, messages, tools, thinking, stream, temperature, fallbacks
    case maxTokens = "max_tokens"
    case toolChoice = "tool_choice"
    case topP = "top_p"
    case topK = "top_k"
    case cacheControl = "cache_control"
    case outputConfig = "output_config"
  }
}

/// Output shaping: structured output via constrained decoding, effort level.
package struct OutputConfig: Sendable, Hashable, Codable {
  package var format: Format?
  package var effort: Effort?

  package init(format: Format? = nil, effort: Effort? = nil) {
    self.format = format
    self.effort = effort
  }

  /// Constrained-decoding output format. The model strictly conforms to the
  /// schema — it cannot emit a token that would violate it — and writes each
  /// object's properties in the order the schema lists them, so the schema
  /// is sent in its own member order (see ``OrderedJSON``).
  package struct Format: Sendable, Hashable, Codable {
    /// The schema as sent, members in order.
    package var orderedSchema: OrderedJSON

    /// The schema's content, member order aside.
    package var schema: JSONValue { orderedSchema.unordered }

    package init(orderedSchema: OrderedJSON) { self.orderedSchema = orderedSchema }

    /// A schema whose member order doesn't matter: objects go out sorted.
    package init(schema: JSONValue) { self.orderedSchema = OrderedJSON(schema) }

    private enum CodingKeys: String, CodingKey { case type, schema }

    package init(from decoder: Decoder) throws {
      let c = try decoder.container(keyedBy: CodingKeys.self)
      orderedSchema = try c.decode(OrderedJSON.self, forKey: .schema)
    }

    package func encode(to encoder: Encoder) throws {
      var c = encoder.container(keyedBy: CodingKeys.self)
      try c.encode("json_schema", forKey: .type)
      try c.encode(orderedSchema, forKey: .schema)
    }
  }

  package enum Effort: String, Sendable, Codable {
    case low, medium, high, xhigh, max
  }
}
