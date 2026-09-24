using System.Text.Json.Nodes;

namespace NLaya;

public sealed record ScoreAnswer : Answer
{
    public override QuestionType Type => QuestionType.Score;

    /// <summary>The expected level, sum(i * p_i).</summary>
    public required double Score { get; init; }

    /// <summary>Level descriptions, index 0 first (Python's <c>legend</c>).</summary>
    public required IReadOnlyList<JsonNode?> Legend { get; init; }

    /// <summary>Probability per level, keyed "0", "1", ...</summary>
    public required OrderedDictionary<string, double> Probabilities { get; init; }

    /// <summary>The single most likely level.</summary>
    public int MostLikelyLevel => Probabilities.Select((kv, i) => (kv.Value, i)).MaxBy(x => x.Value).i;

    public override JsonObject ToJson()
    {
        var legend = new JsonObject();
        for (var i = 0; i < Legend.Count; i++) legend[i.ToString(System.Globalization.CultureInfo.InvariantCulture)] = Legend[i]?.DeepClone();
        var probs = new JsonObject();
        foreach (var (k, v) in Probabilities) probs[k] = v;
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
