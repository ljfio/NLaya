using System.Text.Json.Nodes;

namespace NLaya;

/// <summary>
/// Question id -> <see cref="Question"/>, in order. Collection-initializer friendly:
/// <c>new Questions { ["intent"] = Question.Choice(...), ... }</c>.
/// </summary>
public sealed class Questions : OrderedMap<Question>
{
    public Questions() { }

    public Questions(IEnumerable<KeyValuePair<string, Question>> items) : base(items) { }

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
