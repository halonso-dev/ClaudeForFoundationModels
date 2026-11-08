// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

import ClaudeAPI
import Foundation
import FoundationModels
import Testing

@testable import ClaudeForFoundationModels

/// How a structured request is sent and finished: the schema keeps its
/// declaration order on the wire, optional properties go out required and
/// nullable while the union budget allows, and an answer that didn't end
/// whole fails instead of decoding.
@Suite struct StructuredOutputTests {

  // MARK: - Declaration order on the wire

  @Test func `the schema reaches the wire in declaration order`() async throws {
    let transport = MockTransport(body: turn([.text([Self.wholeAnswer])]))
    let session = LanguageModelSession(model: Self.model(transport))

    _ = try await session.respond(to: "Summarize the days.", generating: DayBatch.self)

    let body = String(decoding: try #require(transport.lastRequest?.httpBody), as: UTF8.self)
    // Order only survives in the raw bytes: each property's schema starts
    // after the one declared before it.
    let offsets = try Self.declared.map { try #require(body.range(of: #""\#($0)":{"#)).lowerBound }
    #expect(offsets == offsets.sorted())
    let required = Self.declared.map { #""\#($0)""# }.joined(separator: ",")
    #expect(body.contains(#""required":[\#(required)]"#))
    // The framework's extension keys still never reach the API.
    #expect(!body.contains("x-order"))
    #expect(!body.contains("\"title\""))
  }

  @Test func `the same schema is sent as the same bytes`() async throws {
    var bodies: [Data] = []
    for _ in 0..<2 {
      let transport = MockTransport(body: turn([.text([Self.wholeAnswer])]))
      let session = LanguageModelSession(model: Self.model(transport))
      _ = try await session.respond(to: "Summarize the days.", generating: DayBatch.self)
      bodies.append(try #require(transport.lastRequest?.httpBody))
    }

    #expect(bodies[0] == bodies[1])
  }

  // A fallback with an effort of its own replaces the whole `output_config`,
  // so it carries a copy of the format, which has to keep the order too.
  // (One without its own effort inherits the request's format as sent.)
  @Test func `a fallback's own output_config keeps the ordered schema`() throws {
    var contextOptions = ContextOptions()
    contextOptions.reasoningLevel = .deep
    let request = LanguageModelExecutorGenerationRequest.make(
      transcript: Transcript(entries: [.prompt(.init(segments: [.text(.init(content: "Hi"))]))]),
      schema: DayBatch.generationSchema,
      contextOptions: contextOptions
    )
    let thinker = ClaudeModel(
      id: "claude-thinker",
      capabilities: .init(effortLevels: [.high], adaptiveThinking: true, structuredOutput: true)
    )
    let steady = ClaudeModel(
      id: "claude-steady",
      capabilities: .init(adaptiveThinking: true, structuredOutput: true)
    )

    let built = try RequestBuilder.build(from: request, model: thinker, fallbacks: [steady])

    let format = try #require(built.request.outputConfig?.format)
    guard case .models(let hops)? = built.request.fallbacks else {
      Issue.record("expected fallback entries")
      return
    }
    // `OrderedJSON` equality includes member order.
    #expect(hops.first?.outputConfig?.format?.orderedSchema == format.orderedSchema)
    #expect(format.orderedSchema == RequestBuilder.outputSchema(from: DayBatch.generationSchema))
  }

  // MARK: - Optional properties

  @Test func `an optional property goes out required and nullable`() throws {
    let schema = RequestBuilder.outputSchema(from: DayEntry.generationSchema).unordered

    #expect(schema["required"] == .array(Self.declared.map(JSONValue.string)))
    // The description moves to the wrapper, where it still reads as the
    // property's.
    #expect(
      schema["properties"]?["milestone"]
        == [
          "anyOf": [["type": "string"], ["type": "null"]],
          "description": "A milestone, or null when there is none.",
        ]
    )
    #expect(
      schema["properties"]?["tags"] == [
        "anyOf": [["type": "array", "items": ["type": "string"]], ["type": "null"]]
      ]
    )
    guard case .array(let detail)? = schema["properties"]?["detail"]?["anyOf"] else {
      Issue.record("expected detail to be nullable")
      return
    }
    #expect(detail.count == 2)
    #expect(detail.last == ["type": "null"])
    #expect(detail.first?["$ref"] != nil)
    // Required properties keep their own shape.
    #expect(schema["properties"]?["recap"] == ["type": "string", "description": "Two sentences."])
  }

  @Test func `null decodes as nil for every optional shape`() async throws {
    let transport = MockTransport(body: turn([.text([Self.wholeAnswer])]))
    let session = LanguageModelSession(model: Self.model(transport))

    let batch = try await session.respond(to: "Summarize the days.", generating: DayBatch.self)
      .content

    let entry = try #require(batch.entries.first)
    #expect(entry.recap == "A calm day.")
    #expect(entry.milestone == nil)
    #expect(entry.detail == nil)
    #expect(entry.tags == nil)
  }

  @Test func `past the union budget, optional properties stay optional`() throws {
    let raw = try #require(JSONValue.encoded(ManyOptionals.generationSchema))
    #expect(RequestBuilder.unionParameterCount(in: raw) == 17)

    let ordered = RequestBuilder.outputSchema(from: ManyOptionals.generationSchema)
    let schema = ordered.unordered

    #expect(schema["required"] == ["name"])
    #expect(schema["properties"]?["o01"] == ["type": "string"])
    // Still in declaration order; the API moves `name` first on its own.
    let text = String(decoding: try ordered.serialized(), as: UTF8.self)
    let names = (1...17).map { String(format: "o%02d", $0) } + ["name"]
    let offsets = try names.map { try #require(text.range(of: #""\#($0)":{"#)).lowerBound }
    #expect(offsets == offsets.sorted())
  }

  @Test func `union parameters are counted through every reference`() throws {
    // Three optionals in DayEntry, reached through the batch's `$ref`.
    let batch = try #require(JSONValue.encoded(DayBatch.generationSchema))
    #expect(RequestBuilder.unionParameterCount(in: batch) == 3)

    // Existing unions count, and a definition counts again at each use.
    let handmade: JSONValue = [
      "type": "object",
      "properties": [
        "a": ["anyOf": [["type": "string"], ["type": "integer"]]],
        "b": ["type": ["string", "null"]],
        "c": ["$ref": "#/$defs/Pair"],
        "d": ["$ref": "#/$defs/Pair"],
      ],
      "required": ["a", "b", "c", "d"],
      "$defs": [
        "Pair": [
          "type": "object",
          "properties": ["x": ["type": "string"], "next": ["$ref": "#/$defs/Pair"]],
          "required": ["next"],
        ]
      ],
    ]
    #expect(RequestBuilder.unionParameterCount(in: handmade) == 4)
  }

  // MARK: - How a structured response ends

  @Test func `the framework alone decodes an answer cut off mid-string as whole`() async throws {
    // Why the stop reason is checked: unchecked, the cut-off answer decodes
    // as if it were whole, holding the cut-off text.
    let transport = MockTransport(body: turn([.text([Self.cutOffAnswer])], stopReason: "end_turn"))
    let session = LanguageModelSession(model: Self.model(transport))

    let batch = try await session.respond(to: "Summarize the days.", generating: DayBatch.self)
      .content

    #expect(batch.entries.first?.whyItMattered == "It set up the")
  }

  @Test func `a structured answer cut off at max_tokens fails instead of decoding`() async throws {
    let transport = MockTransport(
      body: turn([.text([Self.cutOffAnswer])], stopReason: "max_tokens")
    )
    let session = LanguageModelSession(model: Self.model(transport))

    let error = try await #require(throws: LanguageModelError.self) {
      _ = try await session.respond(to: "Summarize the days.", generating: DayBatch.self)
    }

    guard case .contextSizeExceeded(let detail) = error else {
      Issue.record("expected contextSizeExceeded, got \(error)")
      return
    }
    #expect(detail.contextSize == 16_000)
    #expect(detail.tokenCount == 5)
    #expect(!detail.debugDescription.contains("It set up the"))
  }

  @Test func `a structured refusal fails with a refusal`() async throws {
    let transport = MockTransport(body: turn([.text([#"{"entries":["#])], stopReason: "refusal"))
    let session = LanguageModelSession(model: Self.model(transport))

    let error = try await #require(throws: LanguageModelError.self) {
      _ = try await session.respond(to: "Summarize the days.", generating: DayBatch.self)
    }

    guard case .refusal = error else {
      Issue.record("expected refusal, got \(error)")
      return
    }
  }

  @Test func `plain text cut off at max_tokens still comes back`() async throws {
    let transport = MockTransport(body: turn([.text(["Hello, wor"])], stopReason: "max_tokens"))
    let session = LanguageModelSession(model: Self.model(transport))

    let response = try await session.respond(to: "Say hello.")

    #expect(response.content == "Hello, wor")
  }

  @Test func `a plain-text refusal keeps what it streamed`() async throws {
    let transport = MockTransport(
      body: turn([.text(["I can't help with that."])], stopReason: "refusal")
    )
    let session = LanguageModelSession(model: Self.model(transport))

    let response = try await session.respond(to: "Say hello.")

    #expect(response.content == "I can't help with that.")
  }

  // MARK: - Fixtures

  /// A stubbed Claude that takes guided generation, as `ClaudeLanguageModel`
  /// does for a model with structured output.
  private static func model(
    _ transport: MockTransport,
    fallbacks: ClaudeFallbacks = []
  ) -> StubbedClaudeModel {
    StubbedClaudeModel(
      transport: transport,
      capabilities: [.toolCalling, .reasoning, .guidedGeneration],
      fallbacks: fallbacks
    )
  }

  /// `DayEntry`'s properties in declaration order — not alphabetical.
  private static let declared = [
    "dayID", "recap", "topics", "tone", "issues", "whyItMattered", "milestone", "detail", "tags",
  ]

  private static let wholeAnswer = #"""
    {"entries":[{"dayID":"d1","recap":"A calm day.","topics":["work"],"tone":"warm","issues":[],"whyItMattered":"It set up the week.","milestone":null,"detail":null,"tags":null}]}
    """#

  /// Cut inside the last required string, where the app's leaked answers end.
  private static let cutOffAnswer =
    #"{"entries":[{"dayID":"d1","recap":"A calm day.","topics":["work"],"tone":"warm","issues":[],"whyItMattered":"It set up the"#
}

@Generable
private struct DayBatch {
  @Guide(description: "One entry per requested day.")
  var entries: [DayEntry]
}

@Generable
private struct DayEntry {
  var dayID: String
  @Guide(description: "Two sentences.")
  var recap: String
  var topics: [String]
  var tone: DayTone
  var issues: [String]
  var whyItMattered: String
  @Guide(description: "A milestone, or null when there is none.")
  var milestone: String?
  var detail: DayDetail?
  var tags: [String]?
}

@Generable
private enum DayTone { case warm, tense }

@Generable
private struct DayDetail {
  var label: String
}

/// One more optional property than the union budget allows.
@Generable
private struct ManyOptionals {
  var o01: String?
  var o02: String?
  var o03: String?
  var o04: String?
  var o05: String?
  var o06: String?
  var o07: String?
  var o08: String?
  var o09: String?
  var o10: String?
  var o11: String?
  var o12: String?
  var o13: String?
  var o14: String?
  var o15: String?
  var o16: String?
  var o17: String?
  var name: String
}
