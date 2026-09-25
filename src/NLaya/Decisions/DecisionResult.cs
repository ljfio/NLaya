using System.Globalization;
using System.Text.Json.Nodes;

using NLaya.Routing;

namespace NLaya;

/// <summary>
/// A decision with its evidence: port of Python's <c>laya.structured.DecisionResult</c>
/// (<c>decide(..., return_details=True)</c>).
/// </summary>
public class DecisionResult
{
    /// <summary>The schema-shaped values: enum values, integer levels, booleans.</summary>
    public required JsonObject Values { get; init; }

    /// <summary>Each answer's <see cref="Answer.Confidence"/>, keyed by field.</summary>
    public required OrderedDictionary<string, double> Confidence { get; init; }

    /// <summary>Each answer's probabilities, keyed by field. A noul gives <c>{"false": 1 - p, "true": p}</c>.</summary>
    public required OrderedDictionary<string, OrderedDictionary<string, double>> Probabilities { get; init; }

    /// <summary>Laya's raw answer per field.</summary>
    public required OrderedDictionary<string, Answer> Answers { get; init; }

    public required Usage Usage { get; init; }

    /// <summary>Set when a <see cref="Router"/> decided: which checkpoint answered, and why.</summary>
    public RouteDecision? Routing { get; init; }

    internal static DecisionResult Create(JsonObject values, LayaResult result) =>
        new() { Values = values, Confidence = Confidences(result), Probabilities = Probs(result), Answers = result.Answers, Usage = result.Usage, Routing = result.Routing };

    internal static DecisionResult<T> Create<T>(JsonObject values, LayaResult result, T value) =>
        new() { Value = value, Values = values, Confidence = Confidences(result), Probabilities = Probs(result), Answers = result.Answers, Usage = result.Usage, Routing = result.Routing };

    private static OrderedDictionary<string, double> Confidences(LayaResult result)
    {
        var confidence = new OrderedDictionary<string, double>();
        foreach (var (id, answer) in result.Answers) confidence[id] = answer.Confidence;
        return confidence;
    }

    private static OrderedDictionary<string, OrderedDictionary<string, double>> Probs(LayaResult result)
    {
        var probabilities = new OrderedDictionary<string, OrderedDictionary<string, double>>();
        foreach (var (id, answer) in result.Answers)
        {
            probabilities[id] = answer switch
            {
                NoulAnswer n => new() { ["false"] = Calibration.Decoder.Round(1.0 - n.Probability), ["true"] = Calibration.Decoder.Round(n.Probability) },
                ChoiceAnswer c => new(c.Probabilities),
                ScoreAnswer s => new(s.Probabilities.Select((v, i) => KeyValuePair.Create(i.ToString(CultureInfo.InvariantCulture), v))),
                _ => new(),
            };
        }
        return probabilities;
    }

    /// <summary>Python's <c>dataclasses.asdict(result)</c> shape.</summary>
    public JsonObject ToJson()
    {
        var confidence = new JsonObject();
        foreach (var (k, v) in Confidence) confidence[k] = v;
        var probabilities = new JsonObject();
        foreach (var (k, p) in Probabilities)
        {
            var o = new JsonObject();
            foreach (var (label, v) in p) o[label] = v;
            probabilities[k] = o;
        }
        var answers = new JsonObject();
        foreach (var (k, a) in Answers) answers[k] = a.ToJson();
        return new JsonObject
        {
            ["values"] = Values.DeepClone(),
            ["confidence"] = confidence,
            ["probabilities"] = probabilities,
            ["answers"] = answers,
            ["usage"] = Usage.ToJson(),
            ["routing"] = Routing?.ToJson(),
        };
    }
}
