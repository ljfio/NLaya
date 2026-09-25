using System.Text.Json.Serialization;

namespace NLaya.Tests.Decisions;

[JsonSourceGenerationOptions(UseStringEnumConverter = true)]
[JsonSerializable(typeof(LayaTestTicket))]
[JsonSerializable(typeof(TriageTicket))]
[JsonSerializable(typeof(FreeTextTicket))]
internal sealed partial class DecisionTestContext : JsonSerializerContext;
