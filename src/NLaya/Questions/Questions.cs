using System.Text.Json.Nodes;

namespace NLaya;

/// <summary>
/// Question id -> <see cref="Question"/>, in order. Build it fluently,
/// <c>new Questions().Choice("intent", "...", ("refund", "..."), ("other", null)).Noul("urgent", "...")</c>,
/// with a collection initializer, <c>new Questions { ["intent"] = Question.Choice(...) }</c>,
/// or from the Python dict shape with <see cref="Parse"/>.
/// </summary>
/// <remarks>
/// The fluent methods throw on a repeated id, where the indexer (like a Python dict) replaces the
/// earlier question: a repeated id in a chain is almost always a copy-paste mistake.
/// </remarks>
public sealed class Questions : OrderedDictionary<string, Question>
{
    public Questions() { }

    public Questions(IEnumerable<KeyValuePair<string, Question>> items) : base(items) { }

    // ---------------------------------------------------------------- fluent

    /// <summary>Add <paramref name="question"/> as <paramref name="id"/> and return this set.</summary>
    /// <exception cref="ArgumentException"><paramref name="id"/> is already in the set.</exception>
    public Questions With(string id, Question question)
    {
        ArgumentNullException.ThrowIfNull(question);
        if (!TryAdd(id, question)) throw new ArgumentException($"question '{id}' is already defined", nameof(id));
        return this;
    }

    /// <summary>Add a choice between labels, each with an optional description.</summary>
    public Questions Choice(string id, string instructions, params (string Label, string? Description)[] options) =>
        With(id, Question.Choice(instructions, options));

    /// <summary>Add a choice between bare labels.</summary>
    public Questions Choice(string id, string instructions, params string[] labels) =>
        With(id, Question.Choice(instructions, labels));

    /// <summary>Add a choice whose options are added by <paramref name="configure"/>, e.g. in a loop.</summary>
    public Questions Choice(string id, string instructions, Action<ChoiceBuilder> configure)
    {
        ArgumentNullException.ThrowIfNull(configure);
        var b = new ChoiceBuilder();
        configure(b);
        return With(id, b.Build(instructions));
    }

    /// <summary>Add an ordinal score; <paramref name="levels"/> are described from level 0 up.</summary>
    public Questions Score(string id, string instructions, params string[] levels) =>
        With(id, Question.Score(instructions, levels));

    /// <summary>Add an ordinal score whose levels are added by <paramref name="configure"/>, lowest first.</summary>
    public Questions Score(string id, string instructions, Action<ScoreBuilder> configure)
    {
        ArgumentNullException.ThrowIfNull(configure);
        var b = new ScoreBuilder();
        configure(b);
        return With(id, b.Build(instructions));
    }

    /// <summary>Add a yes/no statement; <paramref name="configure"/> sets optional descriptions and labels.</summary>
    public Questions Noul(string id, string instructions, Action<NoulBuilder>? configure = null)
    {
        var b = new NoulBuilder();
        configure?.Invoke(b);
        return With(id, b.Build(instructions));
    }

    // ---------------------------------------------------------------- Python dict shape

    /// <summary>Parse the Python dict shape: <c>{"qid": {"type": ..., "instructions": ..., "criteria": ...}, ...}</c>.</summary>
    public static Questions FromJson(JsonNode? node)
    {
        if (node is not JsonObject o) throw new ArgumentException("questions must be a JSON object of id -> question");
        var qs = new Questions();
        foreach (var (qid, q) in o) qs[qid] = Question.FromJson(q, qid);
        return qs;
    }

    public static Questions Parse(string json) => FromJson(JsonNode.Parse(json));

    public JsonObject ToJson()
    {
        var o = new JsonObject();
        foreach (var (k, q) in this) o[k] = q.ToJson();
        return o;
    }
}
