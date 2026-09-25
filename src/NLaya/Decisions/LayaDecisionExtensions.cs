using System.Diagnostics.CodeAnalysis;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization.Metadata;

namespace NLaya;

/// <summary>
/// <c>Decide</c>: describe the answer as a C# type or JSON schema and get the values back, on
/// <see cref="LayaAgent"/>, <see cref="Routing.Router"/> or any <see cref="ILayaPredictor"/>. Port of
/// Python's <c>decide</c>; see <see cref="DecisionSchema"/> for how properties become questions.
/// </summary>
/// <remarks>
/// For trimmed or native AOT apps, pass a source-generated <see cref="JsonTypeInfo{T}"/> (with
/// <c>UseStringEnumConverter = true</c>); the overloads without one read the type with reflection.
/// </remarks>
public static class LayaDecisionExtensions
{
    // ---------------------------------------------------------------- C# types, source-generated metadata

    /// <summary>Decide a <typeparamref name="T"/> about <paramref name="state"/> in one forward pass.</summary>
    /// <exception cref="LayaSchemaException"><typeparamref name="T"/> has a property Laya can't answer.</exception>
    public static T Decide<T>(this ILayaPredictor predictor, LayaState state, JsonTypeInfo<T> typeInfo, PredictOptions? options = null)
    {
        var schema = DecisionSchema.For(typeInfo);
        return schema.Decide(predictor.Predict(state, schema.Questions, options));
    }

    /// <summary>Decide a <typeparamref name="T"/>, with each field's confidence and probabilities.</summary>
    public static DecisionResult<T> DecideWithDetails<T>(this ILayaPredictor predictor, LayaState state, JsonTypeInfo<T> typeInfo, PredictOptions? options = null)
    {
        var schema = DecisionSchema.For(typeInfo);
        return schema.Details(predictor.Predict(state, schema.Questions, options));
    }

    /// <inheritdoc cref="Decide{T}(ILayaPredictor, LayaState, JsonTypeInfo{T}, PredictOptions?)"/>
    public static async Task<T> DecideAsync<T>(this ILayaPredictor predictor, LayaState state, JsonTypeInfo<T> typeInfo,
        PredictOptions? options = null, CancellationToken ct = default)
    {
        var schema = DecisionSchema.For(typeInfo);
        return schema.Decide(await predictor.PredictAsync(state, schema.Questions, options, ct).ConfigureAwait(false));
    }

    /// <inheritdoc cref="DecideWithDetails{T}(ILayaPredictor, LayaState, JsonTypeInfo{T}, PredictOptions?)"/>
    public static async Task<DecisionResult<T>> DecideWithDetailsAsync<T>(this ILayaPredictor predictor, LayaState state, JsonTypeInfo<T> typeInfo,
        PredictOptions? options = null, CancellationToken ct = default)
    {
        var schema = DecisionSchema.For(typeInfo);
        return schema.Details(await predictor.PredictAsync(state, schema.Questions, options, ct).ConfigureAwait(false));
    }

    // ---------------------------------------------------------------- C# types, reflection

    /// <summary>
    /// Decide a <typeparamref name="T"/> about <paramref name="state"/>, reading <typeparamref name="T"/>
    /// with reflection (C# property names, enums as strings).
    /// </summary>
    [RequiresUnreferencedCode(DecisionSchema.ReflectionMessage)]
    [RequiresDynamicCode(DecisionSchema.ReflectionMessage)]
    public static T Decide<T>(this ILayaPredictor predictor, LayaState state, PredictOptions? options = null) =>
        predictor.Decide(state, DecisionSchema.For<T>().TypeInfo, options);

    /// <inheritdoc cref="Decide{T}(ILayaPredictor, LayaState, PredictOptions?)"/>
    [RequiresUnreferencedCode(DecisionSchema.ReflectionMessage)]
    [RequiresDynamicCode(DecisionSchema.ReflectionMessage)]
    public static DecisionResult<T> DecideWithDetails<T>(this ILayaPredictor predictor, LayaState state, PredictOptions? options = null) =>
        predictor.DecideWithDetails(state, DecisionSchema.For<T>().TypeInfo, options);

    /// <inheritdoc cref="Decide{T}(ILayaPredictor, LayaState, PredictOptions?)"/>
    [RequiresUnreferencedCode(DecisionSchema.ReflectionMessage)]
    [RequiresDynamicCode(DecisionSchema.ReflectionMessage)]
    public static Task<T> DecideAsync<T>(this ILayaPredictor predictor, LayaState state, PredictOptions? options = null, CancellationToken ct = default) =>
        predictor.DecideAsync(state, DecisionSchema.For<T>().TypeInfo, options, ct);

    /// <inheritdoc cref="DecideWithDetails{T}(ILayaPredictor, LayaState, PredictOptions?)"/>
    [RequiresUnreferencedCode(DecisionSchema.ReflectionMessage)]
    [RequiresDynamicCode(DecisionSchema.ReflectionMessage)]
    public static Task<DecisionResult<T>> DecideWithDetailsAsync<T>(this ILayaPredictor predictor, LayaState state,
        PredictOptions? options = null, CancellationToken ct = default) =>
        predictor.DecideWithDetailsAsync(state, DecisionSchema.For<T>().TypeInfo, options, ct);

    // ---------------------------------------------------------------- JSON schemas (Python's decide(runner, state, schema=dict))

    /// <summary>Decide the values of a JSON <paramref name="schema"/> about <paramref name="state"/>.</summary>
    /// <exception cref="LayaSchemaException">The schema can't be expressed as Laya questions.</exception>
    public static JsonObject Decide(this ILayaPredictor predictor, LayaState state, JsonNode schema, PredictOptions? options = null) =>
        predictor.Decide(state, DecisionSchema.FromJson(schema), options);

    /// <inheritdoc cref="Decide(ILayaPredictor, LayaState, JsonNode, PredictOptions?)"/>
    public static JsonObject Decide(this ILayaPredictor predictor, LayaState state, DecisionSchema schema, PredictOptions? options = null) =>
        schema.Project(predictor.Predict(state, schema.Questions, options));

    /// <summary>Decide a JSON schema's values, with each field's confidence and probabilities.</summary>
    public static DecisionResult DecideWithDetails(this ILayaPredictor predictor, LayaState state, JsonNode schema, PredictOptions? options = null) =>
        predictor.DecideWithDetails(state, DecisionSchema.FromJson(schema), options);

    /// <inheritdoc cref="DecideWithDetails(ILayaPredictor, LayaState, JsonNode, PredictOptions?)"/>
    public static DecisionResult DecideWithDetails(this ILayaPredictor predictor, LayaState state, DecisionSchema schema, PredictOptions? options = null) =>
        schema.Details(predictor.Predict(state, schema.Questions, options));

    /// <inheritdoc cref="Decide(ILayaPredictor, LayaState, JsonNode, PredictOptions?)"/>
    public static Task<JsonObject> DecideAsync(this ILayaPredictor predictor, LayaState state, JsonNode schema,
        PredictOptions? options = null, CancellationToken ct = default) =>
        predictor.DecideAsync(state, DecisionSchema.FromJson(schema), options, ct);

    /// <inheritdoc cref="Decide(ILayaPredictor, LayaState, JsonNode, PredictOptions?)"/>
    public static async Task<JsonObject> DecideAsync(this ILayaPredictor predictor, LayaState state, DecisionSchema schema,
        PredictOptions? options = null, CancellationToken ct = default) =>
        schema.Project(await predictor.PredictAsync(state, schema.Questions, options, ct).ConfigureAwait(false));

    /// <inheritdoc cref="DecideWithDetails(ILayaPredictor, LayaState, JsonNode, PredictOptions?)"/>
    public static Task<DecisionResult> DecideWithDetailsAsync(this ILayaPredictor predictor, LayaState state, JsonNode schema,
        PredictOptions? options = null, CancellationToken ct = default) =>
        predictor.DecideWithDetailsAsync(state, DecisionSchema.FromJson(schema), options, ct);

    /// <inheritdoc cref="DecideWithDetails(ILayaPredictor, LayaState, JsonNode, PredictOptions?)"/>
    public static async Task<DecisionResult> DecideWithDetailsAsync(this ILayaPredictor predictor, LayaState state, DecisionSchema schema,
        PredictOptions? options = null, CancellationToken ct = default) =>
        schema.Details(await predictor.PredictAsync(state, schema.Questions, options, ct).ConfigureAwait(false));
}
