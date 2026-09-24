using System.Text.Json.Nodes;

namespace NLaya;

/// <summary>A typed answer. Match on <see cref="ChoiceAnswer"/>, <see cref="ScoreAnswer"/> or <see cref="NoulAnswer"/>.</summary>
public abstract record Answer
{
    public abstract QuestionType Type { get; }

    /// <summary>
    /// Python's <c>confidence</c>: normalized-entropy concentration (1 - H(p)/log k) for choice and
    /// score, and max(p) for noul. Not calibrated for choice/score; gate on
    /// <see cref="AnswerConfidence"/> instead.
    /// </summary>
    public double Confidence { get; init; }

    /// <summary>max(p): the probability of the reported answer, on every type. The calibrated one.</summary>
    public double AnswerConfidence { get; init; }

    public ActionInfo Action { get; init; } = new(0);

    public abstract JsonObject ToJson();
}
