using System.ComponentModel;
using System.ComponentModel.DataAnnotations;

namespace NLaya.Parity;

/// <summary>
/// The C# spelling of <c>decide.json</c>'s schema, minus its integer enum. Each question is its own
/// sequence, so dropping one leaves the others' answers unchanged.
/// </summary>
public sealed record DecideTicket(
    [property: Description("Which team should handle `body`?")] DecideDepartment department,
    [property: Description("How urgent is this?"), Range(1, 3)] int urgency,
    [property: Description("Does the sender ask for money back?")] bool refund_requested,
    bool needs_human);
