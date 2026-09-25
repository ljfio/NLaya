using System.ComponentModel;
using System.ComponentModel.DataAnnotations;

namespace NLaya.Tests.Decisions;

/// <summary>
/// The C# spelling of the fixture's <c>laya_test_schema</c> (minus the integer enum): its questions
/// must equal Python's. Lower-case names so the JSON property names match.
/// </summary>
public sealed record LayaTestTicket(
    [property: Description("Which team?")] Department department,
    [Range(0, 2)] int urgency,
    bool needs_human);
