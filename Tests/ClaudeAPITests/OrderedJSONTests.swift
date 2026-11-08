// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

import Foundation
import Synchronization
import Testing

@testable import ClaudeAPI

/// A structured-output schema reaches the wire in its own member order, while
/// every other object in the body keeps sorted keys. Order only survives in
/// the raw bytes, so these tests read `httpBody` as text.
@Suite struct OrderedJSONTests {

  // MARK: - Serialization

  @Test func `serialized keeps each object's members in order`() throws {
    let value = OrderedJSON.object([
      .init("zeta", .number(1)),
      .init("alpha", .array([.bool(true), .null, .string("a\"b/c")])),
      .init("mid", .object([.init("y", .number(2.5)), .init("x", .bool(false))])),
    ])

    let text = String(decoding: try value.serialized(), as: UTF8.self)

    // Strings are escaped exactly as JSONEncoder escapes them.
    let escaped = String(decoding: try JSONEncoder().encode("a\"b/c"), as: UTF8.self)
    #expect(text == #"{"zeta":1,"alpha":[true,null,\#(escaped)],"mid":{"y":2.5,"x":false}}"#)
  }

  @Test func `a JSONValue converts with sorted members and back unchanged`() throws {
    let value: JSONValue = ["b": [1, "two"], "a": ["d": nil, "c": true]]

    let ordered = OrderedJSON(value)

    #expect(
      String(decoding: try ordered.serialized(), as: UTF8.self)
        == #"{"a":{"c":true,"d":null},"b":[1,"two"]}"#
    )
    #expect(ordered.unordered == value)
  }

  // MARK: - Request bodies

  @Test func `the request body sends the schema in order and sorts everything else`() async throws {
    let transport = BodyRecorder()
    let format = OutputConfig.Format(orderedSchema: Self.schema)
    let request = MessagesRequest(
      model: "m",
      maxTokens: 256,
      messages: [.user("hi")],
      outputConfig: OutputConfig(format: format, effort: .high),
      fallbacks: .models([Fallback(model: "m2", outputConfig: OutputConfig(format: format))])
    )

    _ = try? await client(transport).send(request)
    let body = try #require(transport.bodies.first)
    let text = String(decoding: body, as: UTF8.self)

    // The request's format and the fallback's both keep declaration order.
    let ordered =
      #"{"additionalProperties":false,"properties":{"zeta":{"type":"string"},"alpha":{"type":"integer"}},"required":["zeta","alpha"],"type":"object"}"#
    #expect(text.components(separatedBy: ordered).count == 3)
    // Everything else is still written with sorted keys. (Each pattern
    // matches only the top-level key, not the fallback entry's own.)
    let keys = [
      #"{"fallbacks":["#, #""max_tokens":256"#, #""messages":["#, #""model":"m","#,
      #""output_config":{"effort":"high""#, #""stream":false"#,
    ]
    let offsets = try keys.map { try #require(text.range(of: $0)).lowerBound }
    #expect(offsets == offsets.sorted())
    // And the body is valid JSON with nothing left of the placeholders.
    #expect(!text.contains("raw-json"))
    let decoded = try JSONDecoder().decode(JSONValue.self, from: body)
    #expect(decoded["output_config"]?["format"]?["schema"] == Self.schema.unordered)
  }

  @Test func `the same request encodes to the same bytes every time`() async throws {
    let transport = BodyRecorder()
    let request = MessagesRequest(
      model: "m",
      messages: [.user("hi")],
      outputConfig: OutputConfig(format: .init(orderedSchema: Self.schema))
    )

    _ = try? await client(transport).send(request)
    _ = try? await client(transport).send(request)

    let bodies = transport.bodies
    #expect(bodies.count == 2)
    #expect(bodies.first == bodies.last)
  }

  @Test func `text that looks like a placeholder is left alone`() async throws {
    let transport = BodyRecorder()
    let lookalike = "\u{1}raw-json:token:0\u{1}"
    let request = MessagesRequest(
      model: "m",
      messages: [.user(lookalike)],
      outputConfig: OutputConfig(format: .init(orderedSchema: Self.schema))
    )

    _ = try? await client(transport).send(request)

    let body = try #require(transport.bodies.first)
    let decoded = try JSONDecoder().decode(JSONValue.self, from: body)
    #expect(
      decoded["messages"] == [
        ["role": "user", "content": [["type": "text", "text": .string(lookalike)]]]
      ]
    )
  }

  @Test func `a format encoded outside the client falls back to sorted keys`() throws {
    let format = OutputConfig.Format(orderedSchema: Self.schema)

    let data = try JSONEncoder().encode(format)

    let decoded = try JSONDecoder().decode(OutputConfig.Format.self, from: data)
    #expect(decoded.schema == format.schema)
    #expect(!String(decoding: data, as: UTF8.self).contains("raw-json"))
  }

  // MARK: - Helpers

  /// Declared `zeta` before `alpha`, the reverse of sorted order.
  private static let schema = OrderedJSON.object([
    .init("additionalProperties", .bool(false)),
    .init(
      "properties",
      .object([
        .init("zeta", .object([.init("type", .string("string"))])),
        .init("alpha", .object([.init("type", .string("integer"))])),
      ])
    ),
    .init("required", .array([.string("zeta"), .string("alpha")])),
    .init("type", .string("object")),
  ])

  private func client(_ transport: any HTTPTransport) -> ClaudeClient {
    ClaudeClient(
      configuration: .init(auth: .apiKey("sk-test"), baseURL: URL(string: "https://stub.test")!),
      transport: transport
    )
  }
}

/// Records every request body and answers with an empty 200.
private final class BodyRecorder: HTTPTransport {
  private let recorded = Mutex<[Data]>([])

  var bodies: [Data] { recorded.withLock { $0 } }

  func data(for request: URLRequest) async throws -> (Data, URLResponse) {
    recorded.withLock { $0.append(request.httpBody ?? Data()) }
    return (Data("{}".utf8), response(for: request))
  }

  func bytes(
    for request: URLRequest
  ) async throws -> (AsyncThrowingStream<UInt8, Error>, URLResponse) {
    recorded.withLock { $0.append(request.httpBody ?? Data()) }
    return (AsyncThrowingStream { $0.finish() }, response(for: request))
  }

  private func response(for request: URLRequest) -> URLResponse {
    HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: "HTTP/1.1", headerFields: [:])!
  }
}
