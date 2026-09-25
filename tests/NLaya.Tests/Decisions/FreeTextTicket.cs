namespace NLaya.Tests.Decisions;

/// <summary>A free string can't be a fixed option set.</summary>
public sealed record FreeTextTicket(bool Urgent, string Summary);
