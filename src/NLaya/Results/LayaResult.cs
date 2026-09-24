using System.Text.Json;
using System.Text.Json.Nodes;

namespace NLaya;

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

    /// <summary>Set by <see cref="Routing.Router"/>: which checkpoint answered, and why.</summary>
    public Routing.RouteDecision? Routing { get; private init; }

    internal LayaResult WithRouting(Routing.RouteDecision decision) => Routing is not null ? this : new(Model, Answers, Usage) { Routing = decision };

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
        var o = new JsonObject { ["model"] = Model, ["answers"] = answers, ["usage"] = Usage.ToJson() };
        if (Routing is not null) o["routing"] = Routing.ToJson();
        return o;
    }

    public string ToJsonString(bool indented = false) =>
        ToJson().ToJsonString(new JsonSerializerOptions
        {
            WriteIndented = indented,
            Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
        });

    public override string ToString() => ToJsonString();
}
