using System.ComponentModel;
using System.Text.Json.Serialization;

namespace NLaya.Tests.Decisions;

/// <summary>Enum members with descriptions (the .NET extension) and a custom serialized name.</summary>
public enum Team
{
    [Description("Payments, invoices and refunds")]
    Billing,

    [Description("Bugs and outages")]
    Support,

    [JsonStringEnumMemberName("other_team")]
    Other,
}
