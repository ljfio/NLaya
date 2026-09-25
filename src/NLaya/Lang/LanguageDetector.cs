using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;

namespace NLaya.Lang;

/// <summary>
/// Dependency-free script and language detection used to route between checkpoints: a port of
/// <c>laya.lang</c>. Its word lists and script ranges are embedded from the Python module
/// (<c>lang_data.json</c>). Script detection is exact; the Latin-script language guess is a
/// best-effort stopword/diacritic heuristic.
/// </summary>
public static partial class LanguageDetector
{
    private static readonly LanguageData D = LanguageData.Load();

    // Python's \w is alphanumerics + '_' (no combining marks); these classes spell that out for .NET.
    private const string W = @"[\p{L}\p{N}_]";

    [GeneratedRegex(@"[\p{L}\p{Nl}\p{No}]+")]
    private static partial Regex Word { get; }

    [GeneratedRegex(@"(?<![\p{L}\p{N}_-])[\p{L}\p{N}_-]*(?:[.@][\p{L}\p{N}_-]+)+")]
    private static partial Regex Identifier { get; }

    [GeneratedRegex(@"[=;{}\[\]]|" + W + @"\(")]
    private static partial Regex CodeLine { get; }

    [GeneratedRegex(@"[\p{L}\p{N}][._/\\][\p{L}\p{N}]")]
    private static partial Regex Joined { get; }

    [GeneratedRegex(@"[\p{L}\p{Nl}\p{No}]{2,}")]
    private static partial Regex LetterRun { get; }

    /// <summary>Script and language of a state's text (every string leaf of a JSON state).</summary>
    public static LanguageDetection Analyse(LayaState? state) => Analyse(Leaves(state));

    /// <summary>Script and language of <paramref name="text"/>.</summary>
    public static LanguageDetection Analyse(string text) => Analyse([text]);

    /// <summary>True when the English checkpoint can be expected to read this state.</summary>
    public static bool IsEnglish(LayaState? state) => Analyse(state).IsEnglish;

    /// <summary>Dominant script: "latin", "han", "devanagari", ... or "unknown" when there are no letters.</summary>
    public static string DetectScript(string text) => ScriptFromCounts(ScriptCounts(text));

    /// <summary>Best-effort language code for Latin-script text, or null when undecided.</summary>
    public static string? GuessLatinLanguage(string text) => LatinProfile(text).Language;

    private static LanguageDetection Analyse(List<string> leaves)
    {
        var text = StateText(leaves);
        var counts = ScriptCounts(text);
        var prof = Profile(counts);
        var script = ScriptFromCounts(counts);
        var nonLatin = prof.Count > 0 ? Math.Round(1.0 - (prof.TryGetValue("latin", out var l) ? l : 0.0), 4) : 0.0;
        var nNonLatin = Math.Round(nonLatin * Runes(text).Count(Rune.IsLetter));
        if (script == "latin" && NonLatinWords(text).Count > 0 &&
            (nonLatin >= D.NonLatinFraction || (nonLatin >= D.NonLatinMinFraction && nNonLatin >= D.NonLatinMinLetters)))
            script = prof.Where(kv => kv.Key != "latin").MaxBy(kv => kv.Value).Key; // first of equals, like max()

        if (script == "unknown")
            return new("unknown", prof, null, true, true, 0.0, 0.0, null);
        if (script != "latin")
            return new(script, prof, null, false, true, 0.0, nonLatin, null);

        var lat = LatinProfile(text);
        var lang = lat.Language;
        var undecided = lang is null;
        var english = lang == "en" || (undecided && !lat.LooksNonEnglish);
        string? mixed = null;
        if (english && (leaves.Count > 1 || leaves.Any(x => x.Contains('\n'))) && NonEnglishSegment(leaves) is { } found)
        {
            (lang, mixed) = found;
            english = undecided = false;
        }
        return new("latin", prof, lang, english, undecided, Math.Round(lat.DiacriticRate, 4), nonLatin, mixed);
    }

    // ---------------------------------------------------------------- state text

    private static List<string> Leaves(LayaState? state)
    {
        var out_ = new List<string>();
        if (state?.Text is { } t) out_.Add(t);
        else Collect(state?.Json, 0, out_);
        return out_;
    }

    private static void Collect(JsonNode? n, int depth, List<string> out_)
    {
        if (depth > 6 || n is null) return;
        switch (n)
        {
            case JsonValue v when v.GetValueKind() == JsonValueKind.String: out_.Add(v.GetValue<string>()); break;
            case JsonObject o: foreach (var (_, c) in o) Collect(c, depth + 1, out_); break;
            case JsonArray a: foreach (var c in a) Collect(c, depth + 1, out_); break;
        }
    }

    /// <summary>String leaves joined with spaces, at most <paramref name="maxChars"/> code points.</summary>
    private static string StateText(List<string> leaves, int maxChars = 4000)
    {
        var parts = new List<string>();
        var budget = maxChars;
        foreach (var leaf in leaves)
        {
            if (budget <= 0) break;
            var len = CpLen(leaf);
            if (len > budget)
            {
                parts.Add(CpTake(leaf, budget));
                break;
            }
            parts.Add(leaf);
            budget -= len + 1;
        }
        return CpTake(string.Join(" ", parts), maxChars);
    }

    // ---------------------------------------------------------------- scripts

    private static bool IsLatin(int cp, int limit) =>
        cp < limit || cp is >= 0x1E00 and <= 0x1EFF or >= 0xFF21 and <= 0xFF3A or >= 0xFF41 and <= 0xFF5A;

    private static string? NamedScript(int cp)
    {
        foreach (var (name, ranges) in D.ScriptRanges)
            foreach (var (lo, hi) in ranges)
                if (cp >= lo && cp <= hi) return name;
        return null;
    }

    /// <summary>Letter counts per script in first-seen order, Latin last (it wins no ties).</summary>
    private static List<(string Script, int Count)> ScriptCounts(string text)
    {
        var counts = new List<(string, int)>();
        var latin = 0;
        foreach (var r in Runes(text))
        {
            if (!Rune.IsLetter(r)) continue;
            if (IsLatin(r.Value, 0x02B0)) { latin++; continue; }
            // Letters of scripts with no range are "other": never Latin, never for the English checkpoint.
            var name = NamedScript(r.Value) ?? "other";
            var i = counts.FindIndex(c => c.Item1 == name);
            if (i < 0) counts.Add((name, 1));
            else counts[i] = (name, counts[i].Item2 + 1);
        }
        counts.Add(("latin", latin));
        return counts;
    }

    private static string ScriptFromCounts(List<(string Script, int Count)> counts) =>
        counts.All(c => c.Count == 0) ? "unknown" : counts.MaxBy(c => c.Count).Script;

    private static OrderedDictionary<string, double> Profile(List<(string Script, int Count)> counts)
    {
        var total = counts.Sum(c => c.Count);
        var prof = new OrderedDictionary<string, double>();
        if (total == 0) return prof;
        foreach (var (s, c) in counts.OrderBy(c => c.Script == "latin" ? 0 : 1)) // stable: Latin first
            if (c > 0) prof[s] = (double)c / total;
        return prof;
    }

    /// <summary>Non-Latin words that are not a symbol, a capitalised name or a pronunciation.</summary>
    private static List<string> NonLatinWords(string text)
    {
        var runs = new List<string>();
        var cur = new StringBuilder();
        string? script = null;
        foreach (var r in Runes(text))
        {
            if (Rune.GetUnicodeCategory(r) is UnicodeCategory.NonSpacingMark or UnicodeCategory.EnclosingMark) continue;
            var s = IsLatin(r.Value, 0x0250) ? null : NamedScript(r.Value);
            if (s is not null && s == script)
            {
                cur.Append(r.ToString());
                continue;
            }
            if (cur.Length > 0) runs.Add(cur.ToString());
            cur.Clear();
            script = s;
            if (s is not null) cur.Append(r.ToString());
        }
        if (cur.Length > 0) runs.Add(cur.ToString());
        return runs.Where(w => CpLen(w) >= 2 && !char.IsUpper(w, 0)).ToList();
    }

    // ---------------------------------------------------------------- Latin languages

    private static LatinEvidence LatinProfile(string text)
    {
        var words = Word.Matches(Identifier.Replace(text, " ").Replace("İ", "i").ToLowerInvariant()).Select(m => m.Value).ToList();
        // Python lowercases 'İ' to 'i' + U+0307 (two code points), Unicode's only one-to-many lowercase mapping.
        var lowered = text.Replace("İ", "i\u0307").ToLowerInvariant();
        var diac = lowered.Count(D.Diacritics.Contains);
        var rate = (double)diac / Math.Max(1, CpLen(lowered));
        var nonEnglish = rate >= D.DiacriticRate;
        if (words.Count < 4) return new(null, 0, rate, nonEnglish);

        var scores = D.Stop.Select(s => (s.Lang, Score: words.Count(s.Words.Contains))).ToList();
        var en = scores.First(s => s.Lang == "en").Score;
        // A language may only be named when it matched a word no other list claims.
        var distinct = words.ToHashSet();
        var evidenced = scores.Where(s => s.Lang != "en" &&
            distinct.Any(w => D.Stop.First(x => x.Lang == s.Lang).Words.Contains(w) && !D.Shared.Contains(w))).ToList();
        var (bestLang, best) = evidenced.Count > 0 ? evidenced.MaxBy(s => s.Score) : (null, 0);

        string? lang = null;
        if (bestLang is not null && best >= Math.Max(2, en + 2)) lang = bestLang;
        else if (bestLang is not null && nonEnglish && best >= Math.Max(2, en)) lang = bestLang;
        else if (en > 0 && !nonEnglish) lang = "en";
        return new(lang, en, rate, nonEnglish);
    }

    /// <summary>First line or field that alone reads as a non-English language.</summary>
    private static (string Lang, string Segment)? NonEnglishSegment(List<string> leaves, int maxChars = 4000)
    {
        var seen = 0;
        foreach (var leaf in leaves)
        {
            foreach (var raw in leaf.Split('\n'))
            {
                if (seen >= maxChars) return null;
                var seg = CpTake(raw, maxChars - seen);
                seen += CpLen(seg);
                if (CodeLine.IsMatch(seg)) continue;
                var prose = string.Join(" ", seg.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries).Where(t => !Joined.IsMatch(t)));
                if (prose.Any(char.IsLower))
                    prose = LetterRun.Replace(prose, m => IsAllUpper(m.Value) ? " " : m.Value);
                var tokens = Word.Matches(prose).Select(m => m.Value).ToList();
                if (tokens.Count < 4) continue;
                var lang = LatinProfile(prose).Language;
                if (lang is not null and not "en" &&
                    tokens.Select(t => t.ToLowerInvariant()).Distinct().Count(D.Stop.First(s => s.Lang == lang).Words.Contains) >= 2)
                    return (lang, seg.Trim());
            }
        }
        return null;
    }

    private static bool IsAllUpper(string s) => s.Any(char.IsUpper) && !s.Any(char.IsLower);

    private static StringRuneEnumerator Runes(string s) => s.EnumerateRunes();
    private static int CpLen(string s) => PyStr.Len(s);
    private static string CpTake(string s, int n) => PyStr.Take(s, n);
}
