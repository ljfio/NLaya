using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace NLaya.Calibration;

/// <summary>A per-language temperature override (Python <c>lang_temperatures</c> entry).</summary>
public sealed class LanguageTemperature
{
    /// <summary>Per question type (choice, score, noul). Null inherits the checkpoint's.</summary>
    public IReadOnlyList<double>? Temperature { get; init; }

    /// <summary>Per "type:bucket" key, e.g. "choice:11+". Only these; the checkpoint's buckets are not inherited.</summary>
    public IReadOnlyDictionary<string, double>? TemperatureByOptions { get; init; }
}

/// <summary>Temperature scaling as in <c>laya.common</c>: buckets, clamping and language overrides.</summary>
public sealed class TemperatureTable
{
    /// <summary>Temperatures outside this range distort confidence and are clamped.</summary>
    public const double Min = 0.5, Max = 5.0;

    private readonly double[] _perType;
    private readonly Dictionary<string, double> _byOptions;
    private readonly Dictionary<string, (double[] PerType, Dictionary<string, double> ByOptions)> _lang = new(StringComparer.Ordinal);

    public IReadOnlyList<double> PerType => _perType;
    public IReadOnlyDictionary<string, double> ByOptions => _byOptions;

    /// <summary>Entries that had to be clamped or replaced, formatted like Python's warning.</summary>
    public IReadOnlyList<string> Rejected { get; }

    public TemperatureTable(IReadOnlyList<JsonNode?> perType, IReadOnlyDictionary<string, JsonNode?> byOptions,
        IReadOnlyDictionary<string, LanguageTemperature>? lang = null)
    {
        _perType = perType.Select(Clamp).ToArray();
        if (_perType.Length != 3) throw new ArgumentException("temperature must have 3 entries (choice, score, noul)");
        _byOptions = byOptions.ToDictionary(kv => kv.Key, kv => Clamp(kv.Value), StringComparer.Ordinal);

        var rejected = new List<string>();
        foreach (var (k, raw) in byOptions)
            if (!SameValue(raw, _byOptions[k])) rejected.Add($"{k}={Repr(raw)} -> {_byOptions[k].ToString("G", CultureInfo.InvariantCulture)}");
        for (var i = 0; i < perType.Count; i++)
            if (!SameValue(perType[i], _perType[i])) rejected.Add($"temperature[{i}]={Repr(perType[i])} -> {_perType[i].ToString("G", CultureInfo.InvariantCulture)}");
        Rejected = rejected;

        foreach (var (code, cfg) in lang ?? new Dictionary<string, LanguageTemperature>())
        {
            var norm = NormalizeLang(code);
            var t = cfg.Temperature?.Select(Clamp).ToArray() ?? perType.Select(Clamp).ToArray();
            if (t.Length != 3) throw new ArgumentException($"Language override '{code}' temperature must be a list of 3 floats");
            var tbo = (cfg.TemperatureByOptions ?? new Dictionary<string, double>())
                .ToDictionary(kv => kv.Key, kv => Clamp(kv.Value), StringComparer.Ordinal);
            _lang[norm] = (t, tbo);
        }
    }

    /// <summary>The temperature to divide a question's logits by.</summary>
    public double For(QuestionType type, int optionCount, string? lang = null)
    {
        var bucket = Bucket(type, optionCount);
        if (lang is not null && _lang.TryGetValue(NormalizeLang(lang), out var l))
            return l.ByOptions.TryGetValue(bucket, out var lv) ? lv : l.PerType[(int)type];
        return _byOptions.TryGetValue(bucket, out var v) ? v : _perType[(int)type];
    }

    /// <summary>"choice:2", "score:3-5", "noul:2", "choice:6-10", "choice:11+".</summary>
    public static string Bucket(QuestionType type, int k)
    {
        var size = k <= 2 ? "2" : k <= 5 ? "3-5" : k <= 10 ? "6-10" : "11+";
        return $"{Question.TypeNameOf(type)}:{size}";
    }

    public static double Clamp(double t) => double.IsNaN(t) || double.IsInfinity(t) ? 1.0 : Math.Min(Max, Math.Max(Min, t));

    public static double Clamp(JsonNode? t)
    {
        if (t is JsonValue v)
        {
            if (v.TryGetValue<JsonElement>(out var e))
            {
                if (e.ValueKind == JsonValueKind.Number) return Clamp(e.GetDouble());
                if (e.ValueKind == JsonValueKind.String && double.TryParse(e.GetString(), NumberStyles.Float, CultureInfo.InvariantCulture, out var ds)) return Clamp(ds);
                if (e.ValueKind is JsonValueKind.True) return Clamp(1.0);
                if (e.ValueKind is JsonValueKind.False) return Clamp(0.0);
                return 1.0;
            }
            if (v.TryGetValue<double>(out var d)) return Clamp(d);
            if (v.TryGetValue<int>(out var i)) return Clamp(i);
            if (v.TryGetValue<string>(out var s) && double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out var ps)) return Clamp(ps);
        }
        return 1.0;
    }

    private static bool SameValue(JsonNode? raw, double applied)
    {
        if (raw is JsonValue v && v.GetValueKind() == JsonValueKind.Number) return v.GetValue<double>() == applied;
        return false;
    }

    private static string Repr(JsonNode? n) => n is null ? "None" : PythonJson.Serialize(n);

    internal static string NormalizeLang(string code) => code.Split('-')[0].ToLowerInvariant();
}
