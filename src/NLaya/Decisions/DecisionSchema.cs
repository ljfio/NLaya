using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;
using System.Text.Json.Serialization.Metadata;

namespace NLaya;

/// <summary>
/// A JSON schema planned as Laya questions: port of <c>laya.structured</c>'s
/// <c>plan_from_json_schema</c> / <c>questions_from_json_schema</c> / <c>answers_to_json</c>.
/// Each property becomes one question: an <c>enum</c> or <c>const</c> a choice (all-boolean: a noul),
/// a <c>boolean</c> a noul, and an <c>integer</c>/<c>number</c> with integer <c>minimum</c> and
/// <c>maximum</c> (at most <see cref="MaxScoreLevels"/> levels) a score. Free strings, arrays, nested
/// objects and <c>$ref</c> throw <see cref="LayaSchemaException"/>.
/// </summary>
/// <remarks>
/// .NET extension: a property Python would reject that has <c>oneOf: [{"const": v, "description": "..."}, ...]</c>
/// becomes a choice whose options carry those descriptions. <see cref="For{T}(JsonTypeInfo{T})"/>
/// writes that form for enum members marked <c>[Description]</c>. Schemas Python accepts plan exactly as in Python.
/// </remarks>
public class DecisionSchema
{
    /// <summary>Python's <c>MAX_PROPERTIES</c>.</summary>
    public const int MaxProperties = 32;

    /// <summary>Python's <c>MAX_OPTIONS</c>: most options an enum may have.</summary>
    public const int MaxOptions = 32;

    /// <summary>Python's <c>MAX_SCORE_LEVELS</c>: most levels a bounded number may span.</summary>
    public const int MaxScoreLevels = 10;

    private static readonly ConditionalWeakTable<JsonTypeInfo, DecisionSchema> TypeCache = new();

    private readonly JsonObject _schema;
    private readonly IReadOnlyList<DecisionField> _fields;
    private readonly Questions _questions;

    private protected DecisionSchema(JsonObject schema)
    {
        _schema = schema;
        _fields = DecisionPlanner.Plan(schema);
        _questions = new Questions(_fields.Select(f => KeyValuePair.Create(f.Name, Question.FromJson(f.Question, f.Name))));
    }

    /// <summary>The property names, in order: also the question ids.</summary>
    public IReadOnlyList<string> Fields => _fields.Select(f => f.Name).ToList();

    /// <summary>The questions to ask, one per property. A new collection each time, so callers may edit it.</summary>
    public Questions Questions => new(_questions);

    /// <summary>The JSON schema this was planned from (a copy).</summary>
    public JsonObject ToJsonSchema() => (JsonObject)_schema.DeepClone();

    /// <summary>Plan a JSON schema, as Python's <c>decide(runner, state, schema=dict)</c> does.</summary>
    /// <exception cref="LayaSchemaException">The schema can't be expressed as Laya questions.</exception>
    public static DecisionSchema FromJson(JsonNode? schema) =>
        new((schema as JsonObject)?.DeepClone().AsObject()
            ?? throw new LayaSchemaException($"expected a JSON schema object, got {PyValue.TypeName(schema)}"));

    /// <inheritdoc cref="FromJson(JsonNode?)"/>
    public static DecisionSchema Parse(string json) => FromJson(JsonNode.Parse(json));

    /// <summary>
    /// Plan the schema of <typeparamref name="T"/>, from source-generated (AOT-safe) or reflection metadata.
    /// Enums must serialize as strings. Plans are cached per <paramref name="typeInfo"/>.
    /// </summary>
    /// <exception cref="LayaSchemaException">A property can't be expressed as a Laya question.</exception>
    public static DecisionSchema<T> For<T>(JsonTypeInfo<T> typeInfo)
    {
        ArgumentNullException.ThrowIfNull(typeInfo);
        return (DecisionSchema<T>)TypeCache.GetValue(typeInfo, ti => new DecisionSchema<T>((JsonTypeInfo<T>)ti, TypeSchema.Export(ti)));
    }

    /// <summary>
    /// Plan the schema of <typeparamref name="T"/> with reflection metadata: <paramref name="options"/>,
    /// or defaults that keep C# property names and serialize enums as strings. For trimmed or native AOT
    /// apps, use <see cref="For{T}(JsonTypeInfo{T})"/> with a source-generated context.
    /// </summary>
    [RequiresUnreferencedCode(ReflectionMessage)]
    [RequiresDynamicCode(ReflectionMessage)]
    public static DecisionSchema<T> For<T>(JsonSerializerOptions? options = null)
    {
        options ??= LazyInitializer.EnsureInitialized(ref _reflectionOptions, () =>
        {
            var o = new JsonSerializerOptions { TypeInfoResolver = new DefaultJsonTypeInfoResolver(), Converters = { new JsonStringEnumConverter() } };
            o.MakeReadOnly();
            return o;
        });
        return For((JsonTypeInfo<T>)options.GetTypeInfo(typeof(T)));
    }

    private static JsonSerializerOptions? _reflectionOptions;

    internal const string ReflectionMessage =
        "Reads T's properties with reflection. Pass a JsonTypeInfo<T> from a JsonSerializerContext for trimmed or native AOT apps.";

    /// <summary>
    /// Laya answers (Python's dict shape) to schema values: a choice gives its enum value (with its JSON
    /// type), a score its integer level (the most likely level, not the rounded expected score), a noul
    /// <c>p &gt;= 0.5</c>. Properties without an answer are left out. Port of <c>answers_to_json</c>.
    /// </summary>
    public JsonObject Project(JsonObject answers) => DecisionPlanner.Project(answers, _fields);

    /// <summary>The values <paramref name="result"/> decides.</summary>
    public JsonObject Project(LayaResult result)
    {
        var answers = new JsonObject();
        foreach (var (id, answer) in result.Answers) answers[id] = answer.ToJson();
        return Project(answers);
    }

    /// <summary>The values plus per-field confidence and probabilities (Python's <c>DecisionResult</c>).</summary>
    public DecisionResult Details(LayaResult result) => DecisionResult.Create(Project(result), result);
}
