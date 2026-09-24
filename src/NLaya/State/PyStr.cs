using System.Globalization;
using System.Text;

namespace NLaya;

/// <summary>Python <c>str</c> semantics the ported heuristics depend on: code-point lengths and <c>repr</c>.</summary>
internal static class PyStr
{
    public static int Len(string s)
    {
        var n = 0;
        foreach (var _ in s.EnumerateRunes()) n++;
        return n;
    }

    /// <summary><c>s[:n]</c> in code points.</summary>
    public static string Take(string s, int n)
    {
        if (n <= 0) return "";
        if (s.Length <= n) return s;
        var sb = new StringBuilder();
        foreach (var r in s.EnumerateRunes())
        {
            if (n-- == 0) break;
            sb.Append(r.ToString());
        }
        return sb.ToString();
    }

    /// <summary>Python <c>repr(str)</c>.</summary>
    public static string Repr(string s)
    {
        var q = s.Contains('\'') && !s.Contains('"') ? '"' : '\'';
        var sb = new StringBuilder().Append(q);
        foreach (var r in s.EnumerateRunes())
        {
            var v = r.Value;
            if (v == '\\') sb.Append(@"\\");
            else if (v == q) sb.Append('\\').Append(q);
            else if (v == '\n') sb.Append(@"\n");
            else if (v == '\r') sb.Append(@"\r");
            else if (v == '\t') sb.Append(@"\t");
            else if (Printable(r)) sb.Append(r.ToString());
            else if (v < 0x100) sb.Append(@"\x").Append(v.ToString("x2", CultureInfo.InvariantCulture));
            else if (v < 0x10000) sb.Append(@"\u").Append(v.ToString("x4", CultureInfo.InvariantCulture));
            else sb.Append(@"\U").Append(v.ToString("x8", CultureInfo.InvariantCulture));
        }
        return sb.Append(q).ToString();
    }

    private static bool Printable(Rune r) => r.Value == ' ' || Rune.GetUnicodeCategory(r) is not (
        UnicodeCategory.Control or UnicodeCategory.Format or UnicodeCategory.Surrogate or UnicodeCategory.PrivateUse
        or UnicodeCategory.OtherNotAssigned or UnicodeCategory.LineSeparator or UnicodeCategory.ParagraphSeparator
        or UnicodeCategory.SpaceSeparator);

    /// <summary>Python <c>"%.0f" % x</c>: round half to even.</summary>
    public static string F0(double x) => Math.Round(x, MidpointRounding.ToEven).ToString("0", CultureInfo.InvariantCulture);
}
