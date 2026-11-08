// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

import Foundation
import Synchronization

/// JSON whose objects keep their members in the order they were built in.
///
/// The API's constrained decoding writes an object's properties in the order
/// its schema lists them, so a structured-output schema has to reach the wire
/// in declaration order. ``ClaudeClient`` encodes request bodies with sorted
/// keys to keep their bytes (and the prompt cache) stable, and a keyed
/// container can't hold an order under `.sortedKeys`. So this value writes
/// itself (``serialized()``), and ``ClaudeClient`` splices those bytes into
/// the body in place of a placeholder (see ``RawJSONFragments``).
///
/// Encoded by any other encoder, it falls back to its ``unordered`` content.
package indirect enum OrderedJSON: Sendable, Hashable {
  case null
  case bool(Bool)
  case number(Double)
  case string(String)
  case array([OrderedJSON])
  case object([Member])

  package struct Member: Sendable, Hashable {
    package var key: String
    package var value: OrderedJSON

    package init(_ key: String, _ value: OrderedJSON) {
      self.key = key
      self.value = value
    }
  }
}

extension OrderedJSON {
  /// `value` with every object's members sorted by key.
  package init(_ value: JSONValue) {
    switch value {
    case .null: self = .null
    case .bool(let v): self = .bool(v)
    case .number(let v): self = .number(v)
    case .string(let v): self = .string(v)
    case .array(let v): self = .array(v.map(OrderedJSON.init))
    case .object(let v):
      self = .object(v.keys.sorted().map { Member($0, OrderedJSON(v[$0]!)) })
    }
  }

  /// The same content, member order aside.
  package var unordered: JSONValue {
    switch self {
    case .null: .null
    case .bool(let v): .bool(v)
    case .number(let v): .number(v)
    case .string(let v): .string(v)
    case .array(let v): .array(v.map(\.unordered))
    case .object(let members):
      .object(.init(members.map { ($0.key, $0.value.unordered) }) { first, _ in first })
    }
  }

  /// The JSON text, with every object's members in order. Strings and
  /// numbers go through `JSONEncoder`, so they're escaped and formatted the
  /// way the rest of a request body is.
  package func serialized() throws -> Data {
    var out = Data()
    try write(into: &out, leaves: JSONEncoder())
    return out
  }

  private func write(into out: inout Data, leaves: JSONEncoder) throws {
    switch self {
    case .null:
      out.append(contentsOf: "null".utf8)
    case .bool(let v):
      out.append(contentsOf: (v ? "true" : "false").utf8)
    case .number(let v):
      out.append(try leaves.encode(v))
    case .string(let v):
      out.append(try leaves.encode(v))
    case .array(let items):
      out.append(UInt8(ascii: "["))
      for (index, item) in items.enumerated() {
        if index > 0 { out.append(UInt8(ascii: ",")) }
        try item.write(into: &out, leaves: leaves)
      }
      out.append(UInt8(ascii: "]"))
    case .object(let members):
      out.append(UInt8(ascii: "{"))
      for (index, member) in members.enumerated() {
        if index > 0 { out.append(UInt8(ascii: ",")) }
        out.append(try leaves.encode(member.key))
        out.append(UInt8(ascii: ":"))
        try member.value.write(into: &out, leaves: leaves)
      }
      out.append(UInt8(ascii: "}"))
    }
  }
}

extension OrderedJSON: Codable {
  package init(from decoder: Decoder) throws {
    self.init(try JSONValue(from: decoder))
  }

  package func encode(to encoder: Encoder) throws {
    guard let fragments = encoder.userInfo[RawJSONFragments.userInfoKey] as? RawJSONFragments
    else {
      try unordered.encode(to: encoder)
      return
    }
    var container = encoder.singleValueContainer()
    try container.encode(fragments.placeholder(for: serialized()))
  }
}

/// JSON text written outside `JSONEncoder` during one encode, and the
/// placeholder strings that stand in for it until ``splice(into:using:)``
/// swaps them back. Each placeholder carries a random token, so no other
/// string in the body can match one.
final class RawJSONFragments: Sendable {
  static let userInfoKey = CodingUserInfoKey(rawValue: "ClaudeAPI.RawJSONFragments")!

  private let token = UUID().uuidString
  private let fragments = Mutex<[(placeholder: String, json: Data)]>([])

  func placeholder(for json: Data) -> String {
    fragments.withLock { fragments in
      let placeholder = "\u{1}raw-json:\(token):\(fragments.count)\u{1}"
      fragments.append((placeholder, json))
      return placeholder
    }
  }

  /// `body` with each placeholder, as `encoder` writes it, replaced by its
  /// JSON text. A placeholder that's missing means the body isn't what was
  /// encoded, and sending it would be wrong, so that throws.
  func splice(into body: Data, using encoder: JSONEncoder) throws -> Data {
    var body = body
    for (placeholder, json) in fragments.withLock({ $0 }) {
      let encoded = try encoder.encode(placeholder)
      guard let range = body.range(of: encoded) else {
        throw EncodingError.invalidValue(
          json,
          .init(
            codingPath: [],
            debugDescription: "A raw JSON placeholder is missing from the body."
          )
        )
      }
      body.replaceSubrange(range, with: json)
    }
    return body
  }
}
