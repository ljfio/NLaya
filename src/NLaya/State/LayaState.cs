using System.Diagnostics.CodeAnalysis;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization.Metadata;

namespace NLaya;

/// <summary>
/// What the model reads: a text string, a JSON object, or a conversation (a JSON array, read
/// oldest-first). Implicit conversions cover strings, <see cref="JsonNode"/> and
/// <see cref="JsonElement"/>; use <see cref="From(object?, JsonSerializerOptions?)"/> for POCOs and anonymous objects,
/// or <see cref="From{T}(T, JsonTypeInfo{T})"/> with source-generated metadata under trimming and Native AOT.
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

    internal const string ReflectionMessage =
        "Serializes the state with reflection-based System.Text.Json. For trimming and Native AOT, pass a " +
        "LayaState built with LayaState.From(value, jsonTypeInfo) from a JsonSerializerContext instead.";

    /// <summary>
    /// Build a state from any value: strings stay text, JSON nodes/elements are used as is, and
    /// other objects are serialized with System.Text.Json (property order is kept).
    /// </summary>
    [RequiresUnreferencedCode(ReflectionMessage)]
    [RequiresDynamicCode(ReflectionMessage)]
    public static LayaState From(object? value, JsonSerializerOptions? options = null) =>
        FromKnown(value) ?? FromJson(JsonSerializer.SerializeToNode(value, value?.GetType() ?? typeof(object), options));

    /// <summary>
    /// Build a state from <paramref name="value"/> using source-generated metadata
    /// (<c>MyJsonContext.Default.Ticket</c>): the trimming- and Native AOT-safe form of
    /// <see cref="From(object?, JsonSerializerOptions?)"/>.
    /// </summary>
    public static LayaState From<T>(T value, JsonTypeInfo<T> typeInfo)
    {
        ArgumentNullException.ThrowIfNull(typeInfo);
        return FromKnown(value) ?? FromJson(JsonSerializer.SerializeToNode(value, typeInfo));
    }

    /// <summary>Values that need no serializer: states, strings and JSON.</summary>
    private static LayaState? FromKnown(object? value) => value switch
    {
        LayaState s => s,
        string s => FromText(s),
        JsonNode n => FromJson(n),
        JsonElement e => FromJson(JsonNode.Parse(e.GetRawText())),
        JsonDocument d => FromJson(JsonNode.Parse(d.RootElement.GetRawText())),
        _ => null,
    };

    /// <summary>A conversation from turns, oldest first.</summary>
    [RequiresUnreferencedCode(ReflectionMessage)]
    [RequiresDynamicCode(ReflectionMessage)]
    public static LayaState Conversation(IEnumerable<object?> turns, JsonSerializerOptions? options = null) =>
        Conversation(turns.Select(t => From(t, options)));

    /// <summary>A conversation from turns already built as states (text or JSON), oldest first.</summary>
    public static LayaState Conversation(IEnumerable<LayaState> turns)
    {
        var arr = new JsonArray();
        foreach (var t in turns) arr.Add(t.ToNode());
        return new(null, arr);
    }

    /// <summary>A conversation from turns serialized with source-generated metadata, oldest first.</summary>
    public static LayaState Conversation<T>(IEnumerable<T> turns, JsonTypeInfo<T> typeInfo) =>
        Conversation(turns.Select(t => From(t, typeInfo)));

    /// <summary>The text the model reads, byte-identical to Python <c>laya.common.serialize_state</c>.</summary>
    public string Serialize() => _text ?? PythonJson.Serialize(_json);

    internal JsonNode? ToNode() => _text is not null ? JsonValue.Create(_text) : _json?.DeepClone();

    public static implicit operator LayaState(string text) => FromText(text);
    public static implicit operator LayaState(JsonNode? json) => FromJson(json);
    public static implicit operator LayaState(JsonElement json) => FromJson(JsonNode.Parse(json.GetRawText()));

    public override string ToString() => Serialize();
}
