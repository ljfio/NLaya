using System.Text.Json.Serialization;

namespace NLaya.AotSmoke;

/// <summary>Source-generated JSON metadata: what a Native AOT app passes to <c>LayaState.From</c>.</summary>
[JsonSerializable(typeof(Ticket))]
internal sealed partial class SmokeJsonContext : JsonSerializerContext;
