using System.Text.Json.Serialization;

namespace NLaya.Parity;

[JsonSourceGenerationOptions(UseStringEnumConverter = true, PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower)]
[JsonSerializable(typeof(DecideTicket))]
internal sealed partial class DecideContext : JsonSerializerContext;
