using System.ComponentModel;
using System.ComponentModel.DataAnnotations;

namespace NLaya.Parity;

/// <summary>
/// The C# spelling of <c>decide.json</c>'s schema, minus its integer enum. Each question is its own
/// sequence, so dropping one leaves the others' answers unchanged. <see cref="DecideContext"/> maps the
/// names to Python's snake_case.
/// </summary>
public sealed record DecideTicket(
    [property: Description("Which team should handle `body`?")] DecideDepartment Department,
    [property: Description("How urgent is this?"), Range(1, 3)] int Urgency,
    [property: Description("Does the sender ask for money back?")] bool RefundRequested,
    bool NeedsHuman);
