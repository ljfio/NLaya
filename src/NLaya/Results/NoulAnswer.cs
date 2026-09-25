using System.Text.Json.Nodes;

namespace NLaya;

public sealed record NoulAnswer : Answer
{
    public override QuestionType Type => QuestionType.Noul;

    /// <summary>P(true). Python's <c>noul</c> field.</summary>
    public required double Probability { get; init; }

    /// <summary>The yes/no answer: <see cref="Probability"/> &gt;= 0.5.</summary>
    public bool Value => Probability >= 0.5;

    public override JsonObject ToJson() => new()
    {
        ["type"] = "noul",
        ["noul"] = Probability,
        ["confidence"] = Confidence,
        ["answer_confidence"] = AnswerConfidence,
        ["action"] = Action.ToJson(),
    };
}
