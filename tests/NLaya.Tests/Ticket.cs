namespace NLaya.Tests;

/// <summary>A sample POCO state for comparing reflection and source-generated serialization.</summary>
public sealed record Ticket(string Body, int Id, string[] Tags);
