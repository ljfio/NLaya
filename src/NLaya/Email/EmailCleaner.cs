using System.Text.Json.Nodes;
using System.Text.RegularExpressions;

namespace NLaya.Email;

/// <summary>
/// Strips quoted history, signatures, device footers and disclaimers from an email body so the
/// model reads only the new message. Port of <c>laya.email</c>; its patterns (English, Portuguese,
/// Spanish clients) are embedded from the Python module.
/// </summary>
public static class EmailCleaner
{
    private static readonly Regex[] QuoteHeaders, SignatureMarkers;
    private static readonly Regex AttributionTail, AttributionHead, HeaderFromName, HeaderNext, DeviceFooter, Disclaimer;
    private static readonly Regex Sentence = new(@"(?<=[.!?])\s+", RegexOptions.Compiled);
    private static readonly Regex ParagraphBreak = new(@"\n\s*\n", RegexOptions.Compiled);
    private static readonly Regex Blanks = new(@"[ \t]+", RegexOptions.Compiled);

    static EmailCleaner()
    {
        var d = Resources.Json("Email.email_data.json");
        static Regex R(JsonNode? p) => new(p!["pattern"]!.GetValue<string>(), RegexOptions.Compiled | RegexOptions.CultureInvariant
            | (p["ignore_case"]!.GetValue<bool>() ? RegexOptions.IgnoreCase : RegexOptions.None));
        QuoteHeaders = d["quote_headers"]!.AsArray().Select(R).ToArray();
        SignatureMarkers = d["signature_markers"]!.AsArray().Select(R).ToArray();
        AttributionTail = R(d["attribution_tail"]);
        AttributionHead = R(d["attribution_head"]);
        HeaderFromName = R(d["header_from_name"]);
        HeaderNext = R(d["header_next"]);
        DeviceFooter = R(d["device_footer"]);
        Disclaimer = R(d["disclaimer"]);
    }

    /// <summary>The new message only: no quoted history, signature or disclaimer. Python <c>clean_email_body</c>.</summary>
    public static string CleanBody(string? body, int maxChars = 3000)
    {
        var text = (body ?? "").Replace("\r\n", "\n").Replace("\r", "\n").Replace("\\n", "\n");
        text = PyStr.Take(text, maxChars * 4); // bound regex work; only maxChars are returned

        var src = text.Split('\n');
        var lines = new List<string>();
        for (var i = 0; i < src.Length; i++)
        {
            var line = src[i];
            if (lines.Count > 0 && QuoteHeaders.Any(p => p.IsMatch(line))) break;
            if (lines.Count > 0 && HeaderFromName.IsMatch(line) && i + 1 < src.Length && HeaderNext.IsMatch(src[i + 1])) break;
            if (lines.Count > 0 && AttributionTail.IsMatch(line))
            {
                if (AttributionHead.IsMatch(lines[^1])) lines.RemoveAt(lines.Count - 1);
                break;
            }
            if (line.TrimStart().StartsWith('>')) continue;
            lines.Add(line.TrimEnd());
        }

        var cut = lines.Count;
        for (var i = Math.Max(1, Math.Min((int)(lines.Count * 0.6), lines.Count - 8)); i < lines.Count; i++)
        {
            var n = PyStr.Len(lines[i].Trim());
            if ((n <= 40 && SignatureMarkers.Any(p => p.IsMatch(lines[i]))) || (n <= 60 && DeviceFooter.IsMatch(lines[i])))
            {
                cut = i;
                break;
            }
        }

        var paragraphs = ParagraphBreak.Split(string.Join("\n", lines.Take(cut))).Select(StripDisclaimer);
        var joined = string.Join("\n\n", paragraphs.Select(p => p.Trim()).Where(p => p.Length > 0));
        return PyStr.Take(Blanks.Replace(joined, " "), maxChars);
    }

    /// <summary>A state for email classification: subject, cleaned body, sender and any extra fields. Python <c>email_state</c>.</summary>
    public static JsonObject State(string? subject, string? body, string? sender = null, bool clean = true,
        IEnumerable<KeyValuePair<string, JsonNode?>>? extra = null)
    {
        var state = new JsonObject
        {
            ["subject"] = (subject ?? "").Trim(),
            ["body"] = clean ? CleanBody(body) : body ?? "",
        };
        if (!string.IsNullOrEmpty(sender)) state["from"] = sender;
        foreach (var (k, v) in extra ?? []) if (v is not null) state[k] = v.DeepClone();
        return state;
    }

    /// <summary>Drop disclaimer sentences; a paragraph goes whole only when all of it is boilerplate.</summary>
    private static string StripDisclaimer(string paragraph)
    {
        if (!Disclaimer.IsMatch(paragraph)) return paragraph;
        var pieces = Sentence.Split(paragraph).Select(p => p.Trim()).Where(p => p.Length > 0)
            .SelectMany(p => Disclaimer.IsMatch(p) ? SplitFusedLines(p) : [p]);
        return string.Join(" ", pieces.Where(p => !Disclaimer.IsMatch(p)));
    }

    /// <summary>Split a boilerplate sentence at lines that start a new (uppercase) sentence.</summary>
    private static IEnumerable<string> SplitFusedLines(string sentence)
    {
        if (!sentence.Contains('\n')) return [sentence];
        var pieces = new List<string>();
        var buf = "";
        foreach (var line in sentence.Split('\n').Select(l => l.Trim()).Where(l => l.Length > 0))
        {
            if (buf.Length > 0 && StartsNewSentence(line))
            {
                pieces.Add(buf);
                buf = line;
            }
            else buf = buf.Length > 0 ? buf + " " + line : line;
        }
        if (buf.Length > 0) pieces.Add(buf);
        return pieces;
    }

    private static bool StartsNewSentence(string line)
    {
        foreach (var r in line.EnumerateRunes())
            if (System.Text.Rune.IsLetter(r)) return System.Text.Rune.IsUpper(r);
        return false;
    }
}
