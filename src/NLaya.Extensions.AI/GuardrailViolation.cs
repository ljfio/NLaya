namespace NLaya.Extensions.AI;

/// <summary>
/// One guardrail question that met its threshold. <see cref="Value"/> is P(true) for a noul
/// question and the expected level for a score question.
/// </summary>
public sealed record GuardrailViolation(string QuestionId, QuestionType Type, double Value, double Threshold, double Confidence);
