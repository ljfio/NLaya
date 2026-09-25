using System.ComponentModel;
using System.ComponentModel.DataAnnotations;

namespace NLaya.Tests.Decisions;

public sealed record TriageTicket(
    Team Team,
    [property: Description("How urgent is this?"), Range(1, 3)] int Urgency,
    bool? NeedsHuman,
    [property: Description("Who should it be escalated to, if anyone?")] Team? Escalate);
