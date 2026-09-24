using System.Text.Json.Nodes;

namespace NLaya;

/// <summary>The act/escalate head's output for one question.</summary>
public sealed record ActionInfo(double ActProbability)
{
    public JsonObject ToJson() => new() { ["act_probability"] = ActProbability };
}

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

public sealed record ChoiceAnswer : Answer
{
    public override QuestionType Type => QuestionType.Choice;
    public required string Choice { get; init; }
    public required OrderedMap<double> Probabilities { get; init; }

    public override JsonObject ToJson()
    {
        var probs = new JsonObject();
        foreach (var (k, v) in Probabilities) probs[k] = v;
        return new JsonObject
        {
            ["type"] = "choice",
            ["choice"] = Choice,
            ["probabilities"] = probs,
            ["confidence"] = Confidence,
            ["answer_confidence"] = AnswerConfidence,
            ["action"] = Action.ToJson(),
        };
    }
}

public sealed record ScoreAnswer : Answer
{
    public override QuestionType Type => QuestionType.Score;

    /// <summary>The expected level, sum(i * p_i).</summary>
    public required double Score { get; init; }

    /// <summary>Level descriptions, index 0 first (Python's <c>legend</c>).</summary>
    public required IReadOnlyList<JsonNode?> Legend { get; init; }

    /// <summary>Probability per level, keyed "0", "1", ...</summary>
    public required OrderedMap<double> Probabilities { get; init; }

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
