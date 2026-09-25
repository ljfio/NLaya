using System.Text.Json;
using System.Text.Json.Nodes;

namespace NLaya;

/// <summary>
/// One typed question. Build with <see cref="Choice(string, IEnumerable{KeyValuePair{string, string?}})"/>,
/// <see cref="Score(string, IEnumerable{string})"/> or <see cref="Noul(string, string?, string?, NoulLabels?)"/>,
/// or parse the Python dict shape with <see cref="FromJson(JsonNode?, string)"/>.
/// </summary>
public sealed class Question
{
    private static readonly IReadOnlyDictionary<string, JsonNode?> NoCriteria = System.Collections.ObjectModel.ReadOnlyDictionary<string, JsonNode?>.Empty;

    public QuestionType Type { get; }

    /// <summary>The instructions as given: a string, or structured JSON rendered with Python json.dumps.</summary>
    public JsonNode? Instructions { get; }

    /// <summary>Choice: label -> optional description, in order.</summary>
    public IReadOnlyList<KeyValuePair<string, JsonNode?>> Options { get; }

    /// <summary>Score: level descriptions, index 0 first.</summary>
    public IReadOnlyList<JsonNode?> Levels { get; }

    /// <summary>Noul: optional descriptions keyed "false" / "true".</summary>
    public IReadOnlyDictionary<string, JsonNode?> NoulCriteria { get; }

    /// <summary>Noul: optional display labels.</summary>
    public NoulLabels? Labels { get; }

    internal Question(QuestionType type, JsonNode? instructions, IReadOnlyList<KeyValuePair<string, JsonNode?>>? options,
        IReadOnlyList<JsonNode?>? levels, IReadOnlyDictionary<string, JsonNode?>? noul, NoulLabels? labels)
    {
        Type = type;
        Instructions = instructions;
        Options = options ?? [];
        Levels = levels ?? [];
        NoulCriteria = noul ?? NoCriteria;
        Labels = labels;
    }

    /// <summary>Instruction text the model reads (strings as is, anything else as Python JSON).</summary>
    public string InstructionText => Instructions is JsonValue v && v.GetValueKind() == JsonValueKind.String
        ? v.GetValue<string>()
        : PythonJson.Serialize(Instructions);

    /// <summary>Python's question type name: "choice", "score" or "noul".</summary>
    public string TypeName => TypeNameOf(Type);

    internal static string TypeNameOf(QuestionType t) => t switch
    {
        QuestionType.Choice => "choice",
        QuestionType.Score => "score",
        _ => "noul",
    };

    public int OptionCount => Type switch
    {
        QuestionType.Choice => Options.Count,
        QuestionType.Score => Levels.Count,
        _ => 2,
    };

    // ---------------------------------------------------------------- factories

    /// <summary>A choice between labels, each with an optional description.</summary>
    public static Question Choice(string instructions, IEnumerable<KeyValuePair<string, string?>> criteria) =>
        Checked(new Question(QuestionType.Choice, instructions,
            criteria.Select(kv => new KeyValuePair<string, JsonNode?>(kv.Key, kv.Value is null ? null : JsonValue.Create(kv.Value))).ToList(),
            null, null, null));

    /// <summary>A choice between bare labels.</summary>
    public static Question Choice(string instructions, IEnumerable<string> labels) =>
        Checked(new Question(QuestionType.Choice, instructions,
            labels.Select(l => new KeyValuePair<string, JsonNode?>(l, null)).ToList(), null, null, null));

    /// <summary>A choice between bare labels.</summary>
    public static Question Choice(string instructions, params string[] labels) => Choice(instructions, (IEnumerable<string>)labels);

    /// <summary>A choice between labels, each with an optional description: <c>("refund", "money back"), ("other", null)</c>.</summary>
    public static Question Choice(string instructions, params (string Label, string? Description)[] options) =>
        Choice(instructions, options.Select(o => new KeyValuePair<string, string?>(o.Label, o.Description)));

    /// <summary>An ordinal score; <paramref name="levels"/> are described from index 0 up.</summary>
    public static Question Score(string instructions, IEnumerable<string> levels) =>
        Checked(new Question(QuestionType.Score, instructions, null,
            levels.Select(l => (JsonNode?)JsonValue.Create(l)).ToList(), null, null));

    /// <summary>An ordinal score; <paramref name="levels"/> are described from index 0 up.</summary>
    public static Question Score(string instructions, params string[] levels) => Score(instructions, (IEnumerable<string>)levels);

    /// <summary>A yes/no statement. Descriptions and labels are optional.</summary>
    public static Question Noul(string instructions, string? whenTrue = null, string? whenFalse = null, NoulLabels? labels = null)
    {
        var crit = new OrderedDictionary<string, JsonNode?>();
        if (whenFalse is not null) crit["false"] = JsonValue.Create(whenFalse);
        if (whenTrue is not null) crit["true"] = JsonValue.Create(whenTrue);
        return Checked(new Question(QuestionType.Noul, instructions, null, null, crit, labels));
    }

    private static Question Checked(Question q)
    {
        var err = q.Validate();
        if (err is not null) throw new ArgumentException(err);
        return q;
    }

    // ---------------------------------------------------------------- Python dict shape

    /// <summary>
    /// Parse a question in the Python <c>laya</c> shape, e.g.
    /// <c>{"type": "choice", "instructions": "...", "criteria": {"a": "...", "b": "..."}}</c>.
    /// Errors name the question the way Python's <c>Agent._check_question</c> does.
    /// </summary>
    public static Question FromJson(JsonNode? node, string qid = "?") => QuestionParser.Parse(node, qid);

    /// <summary>The Python dict shape of this question.</summary>
    public JsonObject ToJson()
    {
        var o = new JsonObject { ["type"] = TypeName, ["instructions"] = Instructions?.DeepClone() };
        switch (Type)
        {
            case QuestionType.Choice:
                var c = new JsonObject();
                foreach (var (k, v) in Options) c[k] = v?.DeepClone();
                o["criteria"] = c;
                break;
            case QuestionType.Score:
                o["criteria"] = new JsonArray(Levels.Select(l => l?.DeepClone()).ToArray());
                break;
            default:
                if (NoulCriteria.Count > 0)
                {
                    var n = new JsonObject();
                    foreach (var (k, v) in NoulCriteria) n[k] = v?.DeepClone();
                    o["criteria"] = n;
                }
                if (Labels is not null) o["labels"] = new JsonObject { ["false"] = Labels.False, ["true"] = Labels.True };
                break;
        }
        return o;
    }

    // ---------------------------------------------------------------- validation / rendering

    /// <summary>Null when the question is answerable, else why not.</summary>
    internal string? Validate()
    {
        switch (Type)
        {
            case QuestionType.Choice when Options.Count == 0:
                return "a choice question needs at least one criterion";
            case QuestionType.Choice when Options.Select(o => o.Key).Distinct(StringComparer.Ordinal).Count() != Options.Count:
                return "a choice question's labels must be distinct";
            case QuestionType.Score when Levels.Count == 0:
                return "a score question needs at least one level";
            case QuestionType.Score when Levels.Any(l => l is null):
                return $"score level {Levels.ToList().IndexOf(null)} is null; give every level a description, index 0 first";
            case QuestionType.Noul when Labels is not null && QuestionParser.ParseLabels(new JsonObject { ["false"] = Labels.False, ["true"] = Labels.True }) is null:
                return QuestionParser.LabelsError;
        }
        return null;
    }

    /// <summary>Option texts in label-index order; noul is always [false, true]. Port of <c>render_options</c>.</summary>
    public IReadOnlyList<string> RenderOptions()
    {
        switch (Type)
        {
            case QuestionType.Choice:
                return Options.Select(kv => IsBlank(kv.Value) ? kv.Key : $"{kv.Key}: {RenderCriterion(kv.Value)}").ToList();
            case QuestionType.Score:
                return Levels.Select((c, i) => $"level {i}: {RenderCriterion(c)}").ToList();
            default:
            {
                var falseLabel = Labels?.False.Trim() ?? "false";
                var trueLabel = Labels?.True.Trim() ?? "true";
                NoulCriteria.TryGetValue("false", out var fc);
                NoulCriteria.TryGetValue("true", out var tc);
                return
                [
                    falseLabel + ": " + (IsBlank(fc) ? "no, the statement does not hold" : RenderCriterion(fc)),
                    trueLabel + ": " + (IsBlank(tc) ? "yes, the statement holds" : RenderCriterion(tc)),
                ];
            }
        }
    }

    /// <summary>Only null and "" mean "no description"; 0 and false are real values.</summary>
    private static bool IsBlank(JsonNode? n) =>
        n is null || (n is JsonValue v && v.GetValueKind() == JsonValueKind.String && v.GetValue<string>().Length == 0);

    internal static string RenderCriterion(JsonNode? n) =>
        n is JsonValue v && v.GetValueKind() == JsonValueKind.String ? v.GetValue<string>() : PythonJson.Serialize(n);
}
