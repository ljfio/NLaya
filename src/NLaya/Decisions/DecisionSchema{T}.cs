using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization.Metadata;

namespace NLaya;

/// <summary>
/// The planned schema of <typeparamref name="T"/>: its questions, and how to turn answers into a
/// <typeparamref name="T"/>. Get one from <see cref="DecisionSchema.For{T}(JsonTypeInfo{T})"/>.
/// </summary>
public sealed class DecisionSchema<T> : DecisionSchema
{
    internal DecisionSchema(JsonTypeInfo<T> typeInfo, JsonObject schema) : base(schema) => TypeInfo = typeInfo;

    /// <summary>The serialization metadata values are read with.</summary>
    public JsonTypeInfo<T> TypeInfo { get; }

    /// <summary>Projected values (<see cref="DecisionSchema.Project(LayaResult)"/>) as a <typeparamref name="T"/>.</summary>
    public T Deserialize(JsonObject values) =>
        JsonSerializer.Deserialize(values, TypeInfo) ?? throw new JsonException($"decided values deserialized to null for {typeof(T).Name}");

    /// <summary>The <typeparamref name="T"/> <paramref name="result"/> decides.</summary>
    public T Decide(LayaResult result) => Deserialize(Project(result));

    /// <summary>The <typeparamref name="T"/> <paramref name="result"/> decides, with per-field confidence and probabilities.</summary>
    public new DecisionResult<T> Details(LayaResult result)
    {
        var values = Project(result);
        return DecisionResult.Create(values, result, Deserialize((JsonObject)values.DeepClone()));
    }
}
