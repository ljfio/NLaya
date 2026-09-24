using System.Text.Json;
using System.Text.Json.Nodes;

namespace NLaya;

/// <summary>
/// Reads questions in the Python <c>laya</c> dict shape. Error messages follow Python's
/// <c>Agent._check_question</c>, naming the question and what to fix.
/// </summary>
internal static class QuestionParser
{
    public const string LabelsError = "noul labels must map exactly 'false' and 'true' to distinct non-empty strings";

    public static Question Parse(JsonNode? node, string qid)
    {
        ArgumentException Error(string msg) => new($"question '{qid}': {msg}");

        if (node is not JsonObject o) throw Error($"definition must be a dict, got {PythonTypeName(node)}");
        var type = (o["type"] as JsonValue)?.GetValueKind() == JsonValueKind.String ? o["type"]!.GetValue<string>() : null;
        if (type is not ("choice" or "score" or "noul"))
            throw Error($"unknown type {Repr(o["type"])}; use one of ['choice', 'noul', 'score']");
        if (!o.ContainsKey("instructions")) throw Error("no 'instructions'; add the text the model should answer");
        if (o.ContainsKey("labels") && type != "noul") throw Error("'labels' is only supported for noul questions");

        var instructions = o["instructions"]?.DeepClone();
        var criteria = o["criteria"];
        return type switch
        {
            "choice" => ParseChoice(instructions, criteria, Error),
            "score" => ParseScore(instructions, criteria, Error),
            _ => ParseNoul(instructions, criteria, o, Error),
        };
    }

    /// <summary>Criteria: label -> description, or a list of bare labels.</summary>
    private static Question ParseChoice(JsonNode? instructions, JsonNode? criteria, Func<string, ArgumentException> error)
    {
        List<KeyValuePair<string, JsonNode?>> options = criteria switch
        {
            JsonObject dict => dict.Select(kv => new KeyValuePair<string, JsonNode?>(kv.Key, kv.Value?.DeepClone())).ToList(),
            JsonArray labels => labels.Select(l => new KeyValuePair<string, JsonNode?>(Question.RenderCriterion(l), null)).ToList(),
            _ => throw error("a choice question takes 'criteria' as a dict of label -> description, or a list of labels"),
        };
        if (options.Count == 0) throw error("a choice question needs at least one criterion");
        return new Question(QuestionType.Choice, instructions, options, null, null, null);
    }

    /// <summary>Criteria: level descriptions, index 0 first.</summary>
    private static Question ParseScore(JsonNode? instructions, JsonNode? criteria, Func<string, ArgumentException> error)
    {
        if (criteria is not JsonArray levels)
            throw error("a score question takes 'criteria' as a list of level descriptions, index 0 first");
        if (levels.Count == 0) throw error("a score question needs at least one level");
        for (var i = 0; i < levels.Count; i++)
            if (levels[i] is null) throw error($"score level {i} is null; give every level a description, index 0 first");
        return new Question(QuestionType.Score, instructions, null, levels.Select(l => l?.DeepClone()).ToList(), null, null);
    }

    /// <summary>Criteria (optional): descriptions keyed "true"/"false". Labels (optional): display names for the two options.</summary>
    private static Question ParseNoul(JsonNode? instructions, JsonNode? criteria, JsonObject question, Func<string, ArgumentException> error)
    {
        var descriptions = new OrderedMap<JsonNode?>();
        if (criteria is JsonObject dict)
        {
            var keys = dict.Select(kv => kv.Key.ToLowerInvariant()).ToList();
            if (keys.Any(k => k is not ("true" or "false")))
                throw error(
                    $"a noul question takes 'criteria' keyed only 'true'/'false' (either or both, and omitted is fine), got [{string.Join(", ", keys.Order(StringComparer.Ordinal).Select(PyStr.Repr))}]. " +
                    "Those keys are the option texts the model reads; any other key was silently dropped and replaced with the defaults. " +
                    "If you want the answer worded differently, keep 'criteria' keyed 'true'/'false' and set 'labels' instead.");
            foreach (var (k, v) in dict) descriptions[k.ToLowerInvariant()] = v?.DeepClone();
        }
        else if (criteria is not null)
        {
            throw error("a noul question takes 'criteria' as a dict with optional 'true'/'false' descriptions, or omits it");
        }

        NoulLabels? labels = null;
        if (question.ContainsKey("labels"))
            labels = ParseLabels(question["labels"]) ?? throw error(LabelsError);
        return new Question(QuestionType.Noul, instructions, null, null, descriptions, labels);
    }

    /// <summary>Exactly "false" and "true", mapped to distinct non-empty strings (trimmed); else null.</summary>
    public static NoulLabels? ParseLabels(JsonNode? node)
    {
        if (node is not JsonObject o || o.Count != 2) return null;
        string? Text(string key) => (o[key] as JsonValue)?.GetValueKind() == JsonValueKind.String ? o[key]!.GetValue<string>().Trim() : null;
        var (f, t) = (Text("false"), Text("true"));
        return string.IsNullOrEmpty(f) || string.IsNullOrEmpty(t) || f == t ? null : new NoulLabels(f, t);
    }

    private static string PythonTypeName(JsonNode? n) => n?.GetValueKind() switch
    {
        null or JsonValueKind.Null => "NoneType",
        JsonValueKind.Array => "list",
        JsonValueKind.Object => "dict",
        JsonValueKind.String => "str",
        JsonValueKind.Number => "number",
        _ => "bool",
    };

    private static string Repr(JsonNode? n) =>
        n is null ? "None" : n.GetValueKind() == JsonValueKind.String ? PyStr.Repr(n.GetValue<string>()) : PythonJson.Serialize(n);
}
