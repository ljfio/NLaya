using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Text.Json.Serialization;

namespace NLaya.Tests.Decisions;

/// <summary>
/// The C# spelling of the fixture's <c>laya_test_schema</c> (minus the integer enum): its questions
/// must equal Python's. The JSON names match Python's field names.
/// </summary>
public sealed record LayaTestTicket(
    [property: Description("Which team?"), JsonPropertyName("department")] Department Department,
    [property: JsonPropertyName("urgency")][Range(0, 2)] int Urgency,
    [property: JsonPropertyName("needs_human")] bool NeedsHuman);
