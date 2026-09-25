namespace NLaya.AotSmoke;

/// <summary>A POCO state, serialized through source-generated metadata.</summary>
public sealed record Ticket(string Subject, string Body, string[] Tags);
