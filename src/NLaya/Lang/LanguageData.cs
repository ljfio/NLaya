namespace NLaya.Lang;

/// <summary>
/// The tables <see cref="LanguageDetector"/> runs on: script ranges, stopword lists and diacritics,
/// embedded verbatim from Python's <c>laya.lang</c> (<c>lang_data.json</c>).
/// </summary>
internal sealed record LanguageData(
    List<(string Name, List<(int Lo, int Hi)> Ranges)> ScriptRanges,
    List<(string Lang, HashSet<string> Words)> Stop,
    HashSet<string> Shared,
    HashSet<char> Diacritics,
    double DiacriticRate,
    double NonLatinFraction,
    double NonLatinMinFraction,
    int NonLatinMinLetters)
{
    public static LanguageData Load()
    {
        var o = Resources.Json("Lang.lang_data.json");
        var stop = o["stopwords"]!.AsArray()
            .Select(e => (e![0]!.GetValue<string>(), e[1]!.AsArray().Select(w => w!.GetValue<string>()).ToHashSet(StringComparer.Ordinal)))
            .ToList();
        return new LanguageData(
            o["script_ranges"]!.AsArray().Select(e => (e![0]!.GetValue<string>(),
                e[1]!.AsArray().Select(r => (r![0]!.GetValue<int>(), r[1]!.GetValue<int>())).ToList())).ToList(),
            stop,
            stop.SelectMany(s => s.Item2).GroupBy(w => w).Where(g => g.Count() > 1).Select(g => g.Key).ToHashSet(StringComparer.Ordinal),
            o["non_en_diacritics"]!.GetValue<string>().ToHashSet(),
            o["non_en_diacritic_rate"]!.GetValue<double>(),
            o["non_latin_fraction"]!.GetValue<double>(),
            o["non_latin_min_fraction"]!.GetValue<double>(),
            o["non_latin_min_letters"]!.GetValue<int>());
    }
}
