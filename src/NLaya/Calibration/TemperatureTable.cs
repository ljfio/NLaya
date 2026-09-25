using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace NLaya.Calibration;

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

    /// <summary>True when some language has its own temperatures, so the request language changes answers.</summary>
    internal bool HasLanguageOverrides => _lang.Count > 0;

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

    public static double Clamp(JsonNode? t) => AsDouble(t) is { } d ? Clamp(d) : 1.0;

    /// <summary>Python <c>float(t)</c>: a number, a numeric string or a bool; null for anything else.</summary>
    private static double? AsDouble(JsonNode? t) => t?.GetValueKind() switch
    {
        JsonValueKind.Number => double.Parse(t.ToJsonString(), CultureInfo.InvariantCulture),
        JsonValueKind.String when double.TryParse(t.GetValue<string>(), NumberStyles.Float, CultureInfo.InvariantCulture, out var d) => d,
        JsonValueKind.True => 1.0,
        JsonValueKind.False => 0.0,
        _ => null,
    };

    private static bool SameValue(JsonNode? raw, double applied) => AsDouble(raw) == applied;

    private static string Repr(JsonNode? n) => n is null ? "None" : PythonJson.Serialize(n);

    internal static string NormalizeLang(string code) => code.Split('-')[0].ToLowerInvariant();
}
