using System.Text.Json;
using System.Text.Json.Nodes;

namespace NLaya;

/// <summary>
/// What the model reads: a text string, a JSON object, or a conversation (a JSON array, read
/// oldest-first). Implicit conversions cover strings, <see cref="JsonNode"/> and
/// <see cref="JsonElement"/>; use <see cref="From(object?, JsonSerializerOptions?)"/> for POCOs and anonymous objects.
/// </summary>
public sealed class LayaState
{
    private readonly string? _text;
    private readonly JsonNode? _json;

    private LayaState(string? text, JsonNode? json)
    {
        _text = text;
        _json = json;
    }

    /// <summary>The raw text, when this state is a plain string.</summary>
    public string? Text => _text;

    /// <summary>The JSON value, when this state is structured.</summary>
    public JsonNode? Json => _json;

    /// <summary>
    /// True for a conversation (JSON array). Its newest turns are last, so it is truncated from
    /// the left when it exceeds the token budget.
    /// </summary>
    public bool IsConversation => _json is JsonArray;

    public static LayaState FromText(string text) => new(text ?? throw new ArgumentNullException(nameof(text)), null);

    public static LayaState FromJson(JsonNode? json) => json switch
    {
        JsonValue v when v.GetValueKind() == JsonValueKind.String => new(v.GetValue<string>(), null),
        _ => new(null, json),
    };

    /// <summary>Parse a JSON document into a structured state.</summary>
    public static LayaState ParseJson(string json) => FromJson(JsonNode.Parse(json));

    /// <summary>
    /// Build a state from any value: strings stay text, JSON nodes/elements are used as is, and
    /// other objects are serialized with System.Text.Json (property order is kept).
    /// </summary>
    public static LayaState From(object? value, JsonSerializerOptions? options = null) => value switch
    {
        LayaState s => s,
        string s => FromText(s),
        JsonNode n => FromJson(n),
        JsonElement e => FromJson(JsonNode.Parse(e.GetRawText())),
        JsonDocument d => FromJson(JsonNode.Parse(d.RootElement.GetRawText())),
        _ => FromJson(JsonSerializer.SerializeToNode(value, value?.GetType() ?? typeof(object), options)),
    };

    /// <summary>A conversation from turns, oldest first.</summary>
    public static LayaState Conversation(IEnumerable<object?> turns, JsonSerializerOptions? options = null)
    {
        var arr = new JsonArray();
        foreach (var t in turns)
            arr.Add(t is string s ? JsonValue.Create(s) : From(t, options).ToNode());
        return new(null, arr);
    }

    /// <summary>The text the model reads, byte-identical to Python <c>laya.common.serialize_state</c>.</summary>
    public string Serialize() => _text ?? PythonJson.Serialize(_json);

    internal JsonNode? ToNode() => _text is not null ? JsonValue.Create(_text) : _json?.DeepClone();

    public static implicit operator LayaState(string text) => FromText(text);
    public static implicit operator LayaState(JsonNode? json) => FromJson(json);
    public static implicit operator LayaState(JsonElement json) => From(json);

    public override string ToString() => Serialize();
}
