namespace NLaya.Extensions.AI;

/// <summary>The outcome of one guardrail check: the violations found, and the raw Laya answers.</summary>
public sealed record GuardrailResult(IReadOnlyList<GuardrailViolation> Violations, LayaResult Result)
{
    /// <summary>True when no question met its threshold.</summary>
    public bool Passed => Violations.Count == 0;
}
