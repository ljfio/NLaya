using System.Text.Json.Serialization;

namespace NLaya.Tests;

/// <summary>Source-generated metadata, as a Native AOT app would pass to <c>LayaState.From</c>.</summary>
[JsonSerializable(typeof(Ticket))]
[JsonSerializable(typeof(string))]
internal sealed partial class TestJsonContext : JsonSerializerContext;
