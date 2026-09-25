using System.Text.Json.Nodes;

namespace NLaya;

/// <summary>The answer to a choice question.</summary>
public sealed record ChoiceAnswer : Answer
{
    /// <inheritdoc/>
    public override QuestionType Type => QuestionType.Choice;
    /// <summary>The most probable option's label.</summary>
    public required string Choice { get; init; }
    /// <summary>Probability per option label, in option order.</summary>
    public required OrderedDictionary<string, double> Probabilities { get; init; }

    /// <inheritdoc/>
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
