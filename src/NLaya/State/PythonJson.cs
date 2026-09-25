using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace NLaya;

/// <summary>
/// Serializes JSON exactly like Python's <c>json.dumps(obj, ensure_ascii=False)</c>: <c>", "</c>
/// and <c>": "</c> separators, no HTML escaping, non-ASCII written literally, and floats in
/// Python <c>repr</c> form. The model reads this text, so any byte of difference changes the
/// tokens it sees. System.Text.Json can't be configured to match: it has no spaced separators, even
/// its relaxed encoder escapes emoji and U+2028, and it writes <c>2.0</c> as <c>2</c>.
/// </summary>
internal static class PythonJson
{
    public static string Serialize(JsonNode? node)
    {
        var sb = new StringBuilder();
        Write(node, sb);
        return sb.ToString();
    }

    private static void Write(JsonNode? node, StringBuilder sb)
    {
        switch (node)
        {
            case null:
                sb.Append("null");
                return;
            case JsonObject obj:
            {
                if (obj.Count == 0) { sb.Append("{}"); return; }
                sb.Append('{');
                var first = true;
                foreach (var (k, v) in obj)
                {
                    if (!first) sb.Append(", ");
                    first = false;
                    WriteString(k, sb);
                    sb.Append(": ");
                    Write(v, sb);
                }
                sb.Append('}');
                return;
            }
            case JsonArray arr:
            {
                if (arr.Count == 0) { sb.Append("[]"); return; }
                sb.Append('[');
                for (var i = 0; i < arr.Count; i++)
                {
                    if (i > 0) sb.Append(", ");
                    Write(arr[i], sb);
                }
                sb.Append(']');
                return;
            }
            case JsonValue val:
                WriteValue(val, sb);
                return;
        }
    }

    private static void WriteValue(JsonValue val, StringBuilder sb)
    {
        if (val.TryGetValue<JsonElement>(out var el))
        {
            WriteElement(el, sb);
            return;
        }
        // A CLR value. Floating types are Python floats ("2.0", not "2"); anything else serializes as JSON.
        switch (val.GetValue<object>())
        {
            case double d: sb.Append(FormatFloat(d)); break;
            case float f: sb.Append(FormatFloat(f)); break;
            case decimal m: sb.Append(FormatFloat((double)m)); break;
            default:
                // The node writes itself with its own converter, so no reflection-based serializer is needed.
                using (var doc = JsonDocument.Parse(val.ToJsonString())) WriteElement(doc.RootElement, sb);
                break;
        }
    }

    private static void WriteElement(JsonElement el, StringBuilder sb)
    {
        switch (el.ValueKind)
        {
            case JsonValueKind.Object:
            case JsonValueKind.Array:
                Write(JsonNode.Parse(el.GetRawText()), sb);
                return;
            case JsonValueKind.String: WriteString(el.GetString()!, sb); return;
            case JsonValueKind.True: sb.Append("true"); return;
            case JsonValueKind.False: sb.Append("false"); return;
            case JsonValueKind.Null: sb.Append("null"); return;
            case JsonValueKind.Number:
            {
                var raw = el.GetRawText();
                // json.loads gives an int for integer literals and a float otherwise.
                if (raw.IndexOfAny(['.', 'e', 'E']) < 0) sb.Append(raw);
                else sb.Append(FormatFloat(el.GetDouble()));
                return;
            }
        }
    }

    internal static void WriteString(string s, StringBuilder sb)
    {
        sb.Append('"');
        foreach (var ch in s)
        {
            switch (ch)
            {
                case '"': sb.Append("\\\""); break;
                case '\\': sb.Append("\\\\"); break;
                case '\n': sb.Append("\\n"); break;
                case '\r': sb.Append("\\r"); break;
                case '\t': sb.Append("\\t"); break;
                case '\b': sb.Append("\\b"); break;
                case '\f': sb.Append("\\f"); break;
                default:
                    if (ch < 0x20) sb.Append("\\u").Append(((int)ch).ToString("x4", CultureInfo.InvariantCulture));
                    else sb.Append(ch);
                    break;
            }
        }
        sb.Append('"');
    }

    /// <summary>Python <c>repr(float)</c>: shortest round-trip digits, fixed for 1e-4 &lt;= |x| &lt; 1e16.</summary>
    public static string FormatFloat(double d)
    {
        if (double.IsNaN(d)) return "NaN";
        if (double.IsPositiveInfinity(d)) return "Infinity";
        if (double.IsNegativeInfinity(d)) return "-Infinity";
        if (d == 0) return double.IsNegative(d) ? "-0.0" : "0.0";

        var r = d.ToString("R", CultureInfo.InvariantCulture);
        var neg = r[0] == '-';
        if (neg) r = r[1..];

        string digits;
        int exp; // value = d1.d2d3... * 10^exp
        var ePos = r.IndexOfAny(['E', 'e']);
        if (ePos >= 0)
        {
            var mant = r[..ePos];
            var p = int.Parse(r[(ePos + 1)..], NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture);
            var dot = mant.IndexOf('.');
            var intLen = dot < 0 ? mant.Length : dot;
            digits = mant.Replace(".", "");
            exp = p + intLen - 1;
        }
        else
        {
            var dot = r.IndexOf('.');
            var intPart = dot < 0 ? r : r[..dot];
            var frac = dot < 0 ? "" : r[(dot + 1)..];
            if (intPart.TrimStart('0').Length > 0)
            {
                intPart = intPart.TrimStart('0');
                digits = intPart + frac;
                exp = intPart.Length - 1;
            }
            else
            {
                var z = frac.Length - frac.TrimStart('0').Length;
                digits = frac.TrimStart('0');
                exp = -(z + 1);
            }
        }
        digits = digits.TrimStart('0').TrimEnd('0');
        if (digits.Length == 0) digits = "0";

        var sb = new StringBuilder();
        if (neg) sb.Append('-');
        if (exp is >= -4 and < 16)
        {
            if (exp >= 0)
            {
                var intDigits = digits.Length > exp + 1 ? digits[..(exp + 1)] : digits.PadRight(exp + 1, '0');
                var fracDigits = digits.Length > exp + 1 ? digits[(exp + 1)..] : "0";
                sb.Append(intDigits).Append('.').Append(fracDigits);
            }
            else
            {
                sb.Append("0.").Append('0', -exp - 1).Append(digits);
            }
        }
        else
        {
            sb.Append(digits[0]);
            if (digits.Length > 1) sb.Append('.').Append(digits, 1, digits.Length - 1);
            sb.Append('e').Append(exp < 0 ? '-' : '+').Append(Math.Abs(exp).ToString("00", CultureInfo.InvariantCulture));
        }
        return sb.ToString();
    }
}
