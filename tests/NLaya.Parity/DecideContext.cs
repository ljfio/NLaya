using System.Text.Json.Serialization;

namespace NLaya.Parity;

[JsonSourceGenerationOptions(UseStringEnumConverter = true)]
[JsonSerializable(typeof(DecideTicket))]
internal sealed partial class DecideContext : JsonSerializerContext;
