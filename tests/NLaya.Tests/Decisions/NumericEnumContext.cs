using System.Text.Json.Serialization;

namespace NLaya.Tests.Decisions;

/// <summary>No string enum converter: enums serialize as numbers, which can't be labelled.</summary>
[JsonSerializable(typeof(LayaTestTicket))]
internal sealed partial class NumericEnumContext : JsonSerializerContext;
