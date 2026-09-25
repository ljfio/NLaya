using System.Globalization;
using System.Numerics;
using System.Text.Json.Nodes;

using static System.FormattableString;

namespace NLaya;

/// <summary>
/// Port of <c>laya.structured</c>'s <c>plan_from_json_schema</c> and <c>_project</c>: which schema
/// properties become which questions, and how answers become values. Rules, limits and error messages
/// are Python's, checked against <c>structured.json</c>.
/// </summary>
/// <remarks>
/// One .NET extension: a property Python rejects that has a <c>oneOf</c> of <c>const</c> subschemas
/// (the standard JSON Schema way to describe enum values) becomes a choice whose options carry those
/// descriptions, or a noul when every const is a boolean. Schemas Python accepts plan exactly as in Python.
/// </remarks>
internal static class DecisionPlanner
{
    public static List<DecisionField> Plan(JsonNode? schema)
    {
        if (schema is not JsonObject root)
            throw new LayaSchemaException($"expected a JSON schema object, got {PyValue.TypeName(schema)}");
        var type = root["type"];
        if (!(type is null || PyValue.IsString(type, "object")) || !root.ContainsKey("properties"))
            throw new LayaSchemaException("the top level must be an object with 'properties'");
        if (root["properties"] is not JsonObject properties || properties.Count == 0)
            throw new LayaSchemaException("'properties' must be a non-empty object");
        if (properties.Count > DecisionSchema.MaxProperties)
            throw new LayaSchemaException($"{properties.Count} properties exceeds MAX_PROPERTIES={DecisionSchema.MaxProperties}");
        return properties.Select(kv => Field($"properties.{kv.Key}", kv.Key, kv.Value)).ToList();
    }

    private static DecisionField Field(string path, string name, JsonNode? node)
    {
        if (node is not JsonObject prop)
            throw new LayaSchemaException($"{path}: property must be an object, got {PyValue.TypeName(node)}");
        var description = prop["description"];
        if (prop.ContainsKey("const")) return EnumField(path, name, [(prop["const"], null)], description);
        if (prop.ContainsKey("enum")) return EnumField(path, name, EnumValues(prop["enum"]), description);
        try
        {
            return TypedField(path, name, prop, description);
        }
        catch (LayaSchemaException) when (OneOfConsts(prop) is { } consts)
        {
            return EnumField(path, name, consts, description);
        }
    }

    private static DecisionField TypedField(string path, string name, JsonObject prop, JsonNode? description)
    {
        var jtype = prop["type"];
        if (jtype is JsonArray types) // nullable: ["string", "null"]
            jtype = types.FirstOrDefault(t => !PyValue.IsString(t, "null"));
        if (PyValue.IsString(jtype, "boolean")) return NoulField(name, description, null);
        if (PyValue.IsString(jtype, "string"))
            throw new LayaSchemaException($"{path}: a free string cannot be a fixed option set; use 'enum' or a boolean");
        if (PyValue.IsString(jtype, "integer") || PyValue.IsString(jtype, "number")) return ScoreField(path, name, prop, description);
        if (PyValue.IsString(jtype, "array"))
            throw new LayaSchemaException($"{path}: arrays are not supported; ask one field per element");
        if (PyValue.IsString(jtype, "object"))
            throw new LayaSchemaException($"{path}: nested objects are not supported; flatten the schema");
        if (prop.ContainsKey("$ref"))
            throw new LayaSchemaException($"{path}: $ref/recursion is not supported; flatten the schema");
        throw new LayaSchemaException($"{path}: unsupported schema {PyValue.Repr(prop)}");
    }

    /// <summary>Python iterates whatever <c>enum</c> holds: a list's items, a dict's keys, a string's characters.</summary>
    private static List<(JsonNode? Value, JsonNode? Description)> EnumValues(JsonNode? node) => node switch
    {
        JsonArray a => a.Select(v => (v, (JsonNode?)null)).ToList(),
        JsonObject o => o.Select(kv => ((JsonNode?)kv.Key, (JsonNode?)null)).ToList(),
        JsonValue v when PyValue.TypeName(v) == "str" =>
            PyValue.Str(v).EnumerateRunes().Select(r => ((JsonNode?)r.ToString(), (JsonNode?)null)).ToList(),
        _ => throw new ArgumentException($"object of type '{PyValue.TypeName(node)}' has no len()"),
    };

    /// <summary>The .NET extension: <c>oneOf: [{"const": v, "description": "..."}, ...]</c>, or null.</summary>
    private static List<(JsonNode? Value, JsonNode? Description)>? OneOfConsts(JsonObject prop)
    {
        if (prop["oneOf"] is not JsonArray { Count: > 0 } items || !items.All(i => i is JsonObject o && o.ContainsKey("const")))
            return null;
        return items.Select(i => (i!["const"], i["description"])).ToList();
    }

    private static DecisionField EnumField(string path, string name, List<(JsonNode? Value, JsonNode? Description)> values, JsonNode? description)
    {
        if (values.Count > DecisionSchema.MaxOptions)
            throw new LayaSchemaException($"{path}: {values.Count} options exceeds MAX_OPTIONS={DecisionSchema.MaxOptions}");
        if (values.Count == 0) throw new LayaSchemaException($"{path}: 'enum' must not be empty");
        if (values.All(v => PyValue.IsBool(v.Value))) return NoulField(name, description, values);

        var options = values.Select(v => (Label: v.Value is null ? "null" : PyValue.Str(v.Value), v.Value)).ToList();
        var criteria = new JsonObject();
        // A dict: a repeated label keeps its first position and takes the last description.
        for (var i = 0; i < options.Count; i++) criteria[options[i].Label] = values[i].Description?.DeepClone();
        var question = new JsonObject
        {
            ["type"] = "choice",
            ["instructions"] = Instructions(description, $"What is `{name}`?"),
            ["criteria"] = criteria,
        };
        return new DecisionField(name, QuestionType.Choice, question, options, BigInteger.Zero);
    }

    /// <summary>A boolean field. <paramref name="consts"/> (the oneOf extension) may describe true and false.</summary>
    private static DecisionField NoulField(string name, JsonNode? description, List<(JsonNode? Value, JsonNode? Description)>? consts)
    {
        var question = new JsonObject { ["type"] = "noul", ["instructions"] = Instructions(description, $"Is `{name}` true?") };
        if (consts is not null && consts.Any(c => c.Description is not null))
        {
            var criteria = new JsonObject();
            foreach (var key in new[] { "false", "true" })
                if (consts.LastOrDefault(c => c.Value!.GetValue<bool>() == (key == "true")).Description is { } d)
                    criteria[key] = d.DeepClone();
            question["criteria"] = criteria;
        }
        return new DecisionField(name, QuestionType.Noul, question, [], BigInteger.Zero);
    }

    private static DecisionField ScoreField(string path, string name, JsonObject prop, JsonNode? description)
    {
        if (!PyValue.TryInt(prop["minimum"], out var lo) || !PyValue.TryInt(prop["maximum"], out var hi))
            throw new LayaSchemaException($"{path}: a numeric field needs integer 'minimum' and 'maximum' to become a score");
        if (hi < lo) throw new LayaSchemaException(Invariant($"{path}: 'maximum' {hi} is below 'minimum' {lo}"));
        var span = hi - lo + 1;
        if (span > DecisionSchema.MaxScoreLevels)
            throw new LayaSchemaException(Invariant(
                $"{path}: {span} levels exceeds MAX_SCORE_LEVELS={DecisionSchema.MaxScoreLevels}; narrow the range or use an enum"));
        var levels = new JsonArray();
        for (var v = lo; v <= hi; v++) levels.Add((JsonNode)v.ToString(CultureInfo.InvariantCulture));
        var question = new JsonObject
        {
            ["type"] = "score",
            ["instructions"] = Instructions(description, Invariant($"Score `{name}` from {lo} to {hi}")),
            ["criteria"] = levels,
        };
        return new DecisionField(name, QuestionType.Score, question, [], lo);
    }

    /// <summary><c>description or default</c>: any truthy description, string or not, is the instructions.</summary>
    private static JsonNode? Instructions(JsonNode? description, string fallback) =>
        PyValue.Truthy(description) ? description!.DeepClone() : fallback;

    // ---------------------------------------------------------------- _project

    /// <summary>Answers (Python's dict shape) to schema values: a choice's enum value, a score's integer, a noul's bool.</summary>
    public static JsonObject Project(JsonObject answers, IReadOnlyList<DecisionField> fields)
    {
        var values = new JsonObject();
        foreach (var f in fields)
        {
            if (!answers.TryGetPropertyValue(f.Name, out var node) || node is null) continue;
            var answer = node as JsonObject ?? throw new ArgumentException($"answer '{f.Name}' must be an object");
            switch (f.Kind)
            {
                case QuestionType.Noul:
                    values[f.Name] = PyValue.Float(Get(answer, "noul", 0.0)) >= 0.5;
                    break;
                case QuestionType.Score:
                {
                    var probs = answer["probabilities"];
                    BigInteger idx;
                    if (PyValue.Truthy(probs))
                    {
                        var p = probs as JsonObject ?? throw new ArgumentException($"answer '{f.Name}' probabilities must be an object");
                        // max(range(len(probs)), key=probs[str(i)]): the first most likely level; absent keys count 0.
                        var best = 0;
                        var bestP = double.NegativeInfinity;
                        for (var i = 0; i < p.Count; i++)
                        {
                            var pi = PyValue.Float(Get(p, i.ToString(CultureInfo.InvariantCulture), 0.0));
                            if (pi > bestP) (best, bestP) = (i, pi);
                        }
                        idx = best;
                    }
                    else
                    {
                        idx = new BigInteger(Math.Round(PyValue.Float(Get(answer, "score", 0.0)), MidpointRounding.ToEven)) - f.Minimum;
                    }
                    values[f.Name] = PyValue.Int(f.Minimum + idx);
                    break;
                }
                default:
                {
                    var label = PyValue.Str(answer["choice"]);
                    var match = f.Options.FirstOrDefault(o => o.Label == label);
                    values[f.Name] = match.Label is not null ? match.Value?.DeepClone() : JsonValue.Create(label);
                    break;
                }
            }
        }
        return values;
    }

    /// <summary><c>d.get(key, default)</c>: a present null stays null.</summary>
    private static JsonNode? Get(JsonObject o, string key, double fallback) =>
        o.TryGetPropertyValue(key, out var v) ? v : JsonValue.Create(fallback);
}
