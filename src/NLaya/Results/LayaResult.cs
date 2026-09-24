using System.Text.Json;
using System.Text.Json.Nodes;

namespace NLaya;

/// <summary>Token accounting. Laya never generates, so <see cref="OutputTokens"/> is always 0.</summary>
public sealed record Usage(int InputTokens, int OutputTokens = 0)
{
    public static readonly Usage Zero = new(0, 0);

    public static Usage Sum(IEnumerable<LayaResult> results)
    {
        int i = 0, o = 0;
        foreach (var r in results)
        {
            i += r.Usage.InputTokens;
            o += r.Usage.OutputTokens;
        }
        return new Usage(i, o);
    }

    public JsonObject ToJson() => new() { ["input_tokens"] = InputTokens, ["output_tokens"] = OutputTokens };
}

/// <summary>The answers for one state: the same shape as Python's <c>predict</c> result dict.</summary>
public sealed class LayaResult
{
    public LayaResult(string model, OrderedMap<Answer> answers, Usage usage)
    {
        Model = model;
        Answers = answers;
        Usage = usage;
    }

    public string Model { get; }
    public OrderedMap<Answer> Answers { get; }
    public Usage Usage { get; }

    public Answer this[string questionId] => Answers[questionId];

    /// <summary>The answer to <paramref name="questionId"/> as a specific type.</summary>
    public T Answer<T>(string questionId) where T : Answer => Answers[questionId] as T
        ?? throw new InvalidCastException($"answer '{questionId}' is a {Answers[questionId].Type} answer, not {typeof(T).Name}");

    public ChoiceAnswer Choice(string questionId) => Answer<ChoiceAnswer>(questionId);
    public ScoreAnswer Score(string questionId) => Answer<ScoreAnswer>(questionId);
    public NoulAnswer Noul(string questionId) => Answer<NoulAnswer>(questionId);

    public JsonObject ToJson()
    {
        var answers = new JsonObject();
        foreach (var (k, a) in Answers) answers[k] = a.ToJson();
        return new JsonObject { ["model"] = Model, ["answers"] = answers, ["usage"] = Usage.ToJson() };
    }

    public string ToJsonString(bool indented = false) =>
        ToJson().ToJsonString(new JsonSerializerOptions
        {
            WriteIndented = indented,
            Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
        });

    public override string ToString() => ToJsonString();
}
