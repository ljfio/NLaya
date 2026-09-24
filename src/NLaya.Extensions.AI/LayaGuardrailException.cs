namespace NLaya.Extensions.AI;

/// <summary>Thrown by <see cref="LayaGuardrailChatClient"/> with <see cref="GuardrailAction.Raise"/> (Python's <c>LayaGuardrailError</c>).</summary>
public sealed class LayaGuardrailException(GuardrailResult result)
    : InvalidOperationException($"Laya guardrail policy violation detected: [{string.Join(", ", result.Violations.Select(v => $"'{v.QuestionId}'"))}]")
{
    public GuardrailResult Guardrail { get; } = result;

    public IReadOnlyList<GuardrailViolation> Violations => Guardrail.Violations;

    /// <summary>The raw Laya answers (Python's <c>raw_decision</c>).</summary>
    public LayaResult Result => Guardrail.Result;
}
