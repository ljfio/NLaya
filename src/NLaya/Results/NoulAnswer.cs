using System.Text.Json.Nodes;

namespace NLaya;

public sealed record NoulAnswer : Answer
{
    public override QuestionType Type => QuestionType.Noul;

    /// <summary>P(true).</summary>
    public required double Noul { get; init; }

    /// <summary><see cref="Noul"/> &gt;= 0.5.</summary>
    public bool Value => Noul >= 0.5;

    public override JsonObject ToJson() => new()
    {
        ["type"] = "noul",
        ["noul"] = Noul,
        ["confidence"] = Confidence,
        ["answer_confidence"] = AnswerConfidence,
        ["action"] = Action.ToJson(),
    };
}
