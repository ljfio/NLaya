using System.Text.Json;
using System.Text.Json.Nodes;

namespace NLaya;

/// <summary>The three typed primitives the decision head answers.</summary>
public enum QuestionType
{
    /// <summary>Pick one label from a set.</summary>
    Choice = 0,
    /// <summary>An ordinal level, index 0 first; the answer is the expected level.</summary>
    Score = 1,
    /// <summary>A boolean statement; the answer is P(true).</summary>
    Noul = 2,
}

/// <summary>Display labels for a noul question's two options. Semantics stay false/true.</summary>
public sealed record NoulLabels(string False, string True);

/// <summary>
/// One typed question. Build with <see cref="Choice(string, IEnumerable{KeyValuePair{string, string?}})"/>,
/// <see cref="Score(string, IEnumerable{string})"/> or <see cref="Noul(string, string?, string?, NoulLabels?)"/>,
/// or parse the Python dict shape with <see cref="FromJson(JsonNode?, string)"/>.
/// </summary>
public sealed class Question
{
    private static readonly OrderedMap<JsonNode?> NoCriteria = new();

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

    private Question(QuestionType type, JsonNode? instructions, IReadOnlyList<KeyValuePair<string, JsonNode?>>? options,
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

    /// <summary>An ordinal score; <paramref name="levels"/> are described from index 0 up.</summary>
    public static Question Score(string instructions, IEnumerable<string> levels) =>
        Checked(new Question(QuestionType.Score, instructions, null,
            levels.Select(l => (JsonNode?)JsonValue.Create(l)).ToList(), null, null));

    /// <summary>An ordinal score; <paramref name="levels"/> are described from index 0 up.</summary>
    public static Question Score(string instructions, params string[] levels) => Score(instructions, (IEnumerable<string>)levels);

    /// <summary>A yes/no statement. Descriptions and labels are optional.</summary>
    public static Question Noul(string instructions, string? whenTrue = null, string? whenFalse = null, NoulLabels? labels = null)
    {
        var crit = new OrderedMap<JsonNode?>();
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
    public static Question FromJson(JsonNode? node, string qid = "?")
    {
        string Err(string msg) => $"question '{qid}': {msg}";
        if (node is not JsonObject o)
            throw new ArgumentException(Err($"definition must be a dict, got {JsonKind(node)}"));
        var typeName = o["type"] is JsonValue tv && tv.GetValueKind() == JsonValueKind.String ? tv.GetValue<string>() : null;
        QuestionType type = typeName switch
        {
            "choice" => QuestionType.Choice,
            "score" => QuestionType.Score,
            "noul" => QuestionType.Noul,
            _ => throw new ArgumentException(Err($"unknown type {PyRepr(o["type"])}; use one of ['choice', 'noul', 'score']")),
        };
        if (!o.ContainsKey("instructions"))
            throw new ArgumentException(Err("no 'instructions'; add the text the model should answer"));
        var ins = o["instructions"]?.DeepClone();
        var crit = o["criteria"];
        if (o.ContainsKey("labels") && type != QuestionType.Noul)
            throw new ArgumentException(Err("'labels' is only supported for noul questions"));

        switch (type)
        {
            case QuestionType.Choice:
            {
                List<KeyValuePair<string, JsonNode?>> opts = crit switch
                {
                    JsonObject co => co.Select(kv => new KeyValuePair<string, JsonNode?>(kv.Key, kv.Value?.DeepClone())).ToList(),
                    JsonArray ca => ca.Select(c => new KeyValuePair<string, JsonNode?>(LabelText(c), null)).ToList(),
                    _ => throw new ArgumentException(Err("a choice question takes 'criteria' as a dict of label -> description, or a list of labels")),
                };
                if (opts.Count == 0) throw new ArgumentException(Err("a choice question needs at least one criterion"));
                return new Question(type, ins, opts, null, null, null);
            }
            case QuestionType.Score:
            {
                if (crit is not JsonArray levels)
                    throw new ArgumentException(Err("a score question takes 'criteria' as a list of level descriptions, index 0 first"));
                if (levels.Count == 0) throw new ArgumentException(Err("a score question needs at least one level"));
                var nullAt = levels.Select((l, i) => (l, i)).FirstOrDefault(x => x.l is null, (null, -1)).Item2;
                if (nullAt >= 0)
                    throw new ArgumentException(Err($"score level {nullAt} is null; give every level a description, index 0 first"));
                return new Question(type, ins, null, levels.Select(l => l?.DeepClone()).ToList(), null, null);
            }
            default:
            {
                var map = new OrderedMap<JsonNode?>();
                if (crit is JsonObject no)
                {
                    var keys = no.Select(kv => kv.Key.ToLowerInvariant()).ToList();
                    if (keys.Any(k => k is not ("true" or "false")))
                        throw new ArgumentException(Err(
                            $"a noul question takes 'criteria' keyed only 'true'/'false' (either or both, and omitted is fine), got [{string.Join(", ", keys.Order(StringComparer.Ordinal).Select(k => $"'{k}'"))}]. " +
                            "Those keys are the option texts the model reads; any other key was silently dropped and replaced with the defaults. " +
                            "If you want the answer worded differently, keep 'criteria' keyed 'true'/'false' and set 'labels' instead."));
                    foreach (var (k, v) in no) map[k.ToLowerInvariant()] = v?.DeepClone();
                }
                else if (crit is not null)
                {
                    throw new ArgumentException(Err("a noul question takes 'criteria' as a dict with optional 'true'/'false' descriptions, or omits it"));
                }
                NoulLabels? labels = null;
                if (o.ContainsKey("labels"))
                {
                    labels = ParseLabels(o["labels"]) ?? throw new ArgumentException(Err(LabelsError));
                }
                return new Question(type, ins, null, null, map, labels);
            }
        }
    }

    private const string LabelsError = "noul labels must map exactly 'false' and 'true' to distinct non-empty strings";

    private static NoulLabels? ParseLabels(JsonNode? node)
    {
        if (node is not JsonObject lo || lo.Count != 2) return null;
        if (lo["false"] is not JsonValue f || f.GetValueKind() != JsonValueKind.String) return null;
        if (lo["true"] is not JsonValue t || t.GetValueKind() != JsonValueKind.String) return null;
        var fs = f.GetValue<string>().Trim();
        var ts = t.GetValue<string>().Trim();
        return fs.Length == 0 || ts.Length == 0 || fs == ts ? null : new NoulLabels(fs, ts);
    }

    private static string LabelText(JsonNode? n) => n is JsonValue v && v.GetValueKind() == JsonValueKind.String
        ? v.GetValue<string>()
        : PythonJson.Serialize(n);

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
            case QuestionType.Noul when Labels is not null && ParseLabels(new JsonObject { ["false"] = Labels.False, ["true"] = Labels.True }) is null:
                return LabelsError;
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

    private static string JsonKind(JsonNode? n) => n switch
    {
        null => "NoneType",
        JsonArray => "list",
        JsonObject => "dict",
        JsonValue v => v.GetValueKind() switch
        {
            JsonValueKind.String => "str",
            JsonValueKind.Number => "number",
            _ => "bool",
        },
        _ => "unknown",
    };

    private static string PyRepr(JsonNode? n) => n is JsonValue v && v.GetValueKind() == JsonValueKind.String
        ? $"'{v.GetValue<string>()}'"
        : n is null ? "None" : PythonJson.Serialize(n);
}
