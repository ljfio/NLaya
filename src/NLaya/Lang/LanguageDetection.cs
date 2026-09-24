using System.Text.Json.Nodes;

namespace NLaya.Lang;

/// <summary>Result of <see cref="LanguageDetector.Analyse(LayaState?)"/>; same fields as Python's <c>laya.lang.analyse</c>.</summary>
public sealed record LanguageDetection(
    string Script,
    OrderedMap<double> ScriptProfile,
    string? Language,
    bool IsEnglish,
    bool LanguageUndecided,
    double DiacriticRate,
    double NonLatinFraction,
    string? MixedSegment)
{
    public JsonObject ToJson()
    {
        var prof = new JsonObject();
        foreach (var (k, v) in ScriptProfile) prof[k] = v;
        return new JsonObject
        {
            ["script"] = Script,
            ["script_profile"] = prof,
            ["language"] = Language,
            ["is_english"] = IsEnglish,
            ["language_undecided"] = LanguageUndecided,
            ["diacritic_rate"] = DiacriticRate,
            ["non_latin_fraction"] = NonLatinFraction,
            ["mixed_segment"] = MixedSegment,
        };
    }
}
