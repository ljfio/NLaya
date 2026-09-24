namespace NLaya.Routing;

/// <summary>Which checkpoint a request goes to, and why.</summary>
public sealed record RouteDecision(string Model, string Reason, string? Lang = null, string? Script = null);
