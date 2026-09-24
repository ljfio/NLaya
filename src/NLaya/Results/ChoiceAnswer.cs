using System.Text.Json.Nodes;

namespace NLaya;

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
