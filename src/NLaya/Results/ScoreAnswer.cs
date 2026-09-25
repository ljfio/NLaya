using System.Text.Json.Nodes;

namespace NLaya;

/// <summary>The answer to an ordinal score question.</summary>
public sealed record ScoreAnswer : Answer
{
    /// <inheritdoc/>
    public override QuestionType Type => QuestionType.Score;

    /// <summary>The expected level, sum(i * p_i).</summary>
    public required double Score { get; init; }

    /// <summary>Level descriptions, index 0 first (Python's <c>legend</c>).</summary>
    public required IReadOnlyList<JsonNode?> Legend { get; init; }

    /// <summary>Probability per level, level 0 first. <see cref="ToJson"/> keys them "0", "1", ... as Python does.</summary>
    public required IReadOnlyList<double> Probabilities { get; init; }

    /// <summary>The single most likely level.</summary>
    public int MostLikelyLevel => Probabilities.Count == 0 ? 0 : Probabilities.Index().MaxBy(x => x.Item).Index;

    /// <inheritdoc/>
    public override JsonObject ToJson()
    {
        var legend = new JsonObject();
        for (var i = 0; i < Legend.Count; i++) legend[i.ToString(System.Globalization.CultureInfo.InvariantCulture)] = Legend[i]?.DeepClone();
        var probs = new JsonObject();
        for (var i = 0; i < Probabilities.Count; i++) probs[i.ToString(System.Globalization.CultureInfo.InvariantCulture)] = Probabilities[i];
        return new JsonObject
        {
            ["type"] = "score",
            ["score"] = Score,
            ["legend"] = legend,
            ["probabilities"] = probs,
            ["confidence"] = Confidence,
            ["answer_confidence"] = AnswerConfidence,
            ["action"] = Action.ToJson(),
        };
    }
}
