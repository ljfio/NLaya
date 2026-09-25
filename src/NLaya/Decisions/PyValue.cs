using System.Globalization;
using System.Numerics;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace NLaya;

/// <summary>
/// Python semantics for values as <c>json.loads</c> returns them, which <c>laya.structured</c> relies
/// on: <c>type(v).__name__</c>, <c>str</c>, <c>repr</c>, truthiness, and <c>isinstance(v, int)</c>
/// (bool included). Integer literals are Python ints and anything with <c>.</c> or an exponent is a float.
/// </summary>
internal static class PyValue
{
    private enum Kind { None, Bool, Int, Float, Str, List, Dict }

    private static Kind KindOf(JsonNode? n) => n switch
    {
        null => Kind.None,
        JsonObject => Kind.Dict,
        JsonArray => Kind.List,
        JsonValue v => v.GetValueKind() switch
        {
            JsonValueKind.String => Kind.Str,
            JsonValueKind.True or JsonValueKind.False => Kind.Bool,
            JsonValueKind.Number => IsIntegral(v) ? Kind.Int : Kind.Float,
            _ => Kind.None,
        },
        _ => Kind.None,
    };

    // Parsed JSON: an integer literal. A CLR value: any number that isn't a floating type.
    private static bool IsIntegral(JsonValue v) => v.TryGetValue<JsonElement>(out var e)
        ? e.GetRawText().IndexOfAny(['.', 'e', 'E']) < 0
        : v.GetValue<object>() is not (double or float or decimal or Half);

    private static string Text(JsonNode n) =>
        n.AsValue().TryGetValue<string>(out var s) ? s : JsonNode.Parse(n.ToJsonString())!.GetValue<string>();

    /// <summary><c>type(v).__name__</c>.</summary>
    public static string TypeName(JsonNode? n) => KindOf(n) switch
    {
        Kind.None => "NoneType",
        Kind.Bool => "bool",
        Kind.Int => "int",
        Kind.Float => "float",
        Kind.Str => "str",
        Kind.List => "list",
        _ => "dict",
    };

    /// <summary><c>isinstance(v, int)</c>, which is true for bools too.</summary>
    public static bool TryInt(JsonNode? n, out BigInteger value)
    {
        value = default;
        switch (KindOf(n))
        {
            case Kind.Bool:
                value = n!.GetValue<bool>() ? BigInteger.One : BigInteger.Zero;
                return true;
            case Kind.Int:
                value = BigInteger.Parse(n!.ToJsonString(), NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture);
                return true;
            default:
                return false;
        }
    }

    public static bool IsBool(JsonNode? n) => KindOf(n) == Kind.Bool;

    public static bool IsString(JsonNode? n, string s) => KindOf(n) == Kind.Str && Text(n!) == s;

    /// <summary><c>bool(v)</c>.</summary>
    public static bool Truthy(JsonNode? n)
    {
        switch (KindOf(n))
        {
            case Kind.None: return false;
            case Kind.Bool: return n!.GetValue<bool>();
            case Kind.Int:
                TryInt(n, out var i);
                return !i.IsZero;
            case Kind.Float: return Float(n) != 0.0;
            case Kind.Str: return Text(n!).Length > 0;
            case Kind.List: return ((JsonArray)n!).Count > 0;
            default: return ((JsonObject)n!).Count > 0;
        }
    }

    /// <summary><c>float(v)</c> for numbers, bools and numeric strings.</summary>
    public static double Float(JsonNode? n) => KindOf(n) switch
    {
        Kind.Bool => n!.GetValue<bool>() ? 1.0 : 0.0,
        Kind.Int or Kind.Float => double.Parse(n!.ToJsonString(), NumberStyles.Float, CultureInfo.InvariantCulture),
        Kind.Str when double.TryParse(Text(n!).Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out var d) => d,
        _ => throw new ArgumentException($"float() argument must be a string or a real number, not '{TypeName(n)}'"),
    };

    /// <summary><c>str(v)</c>.</summary>
    public static string Str(JsonNode? n) => KindOf(n) == Kind.Str ? Text(n!) : Repr(n);

    /// <summary><c>repr(v)</c>.</summary>
    public static string Repr(JsonNode? n)
    {
        var sb = new StringBuilder();
        WriteRepr(n, sb);
        return sb.ToString();
    }

    private static void WriteRepr(JsonNode? n, StringBuilder sb)
    {
        switch (KindOf(n))
        {
            case Kind.None: sb.Append("None"); break;
            case Kind.Bool: sb.Append(n!.GetValue<bool>() ? "True" : "False"); break;
            case Kind.Int:
                TryInt(n, out var i);
                sb.Append(i.ToString(CultureInfo.InvariantCulture));
                break;
            case Kind.Float: sb.Append(PythonJson.FormatFloat(Float(n))); break;
            case Kind.Str: sb.Append(PyStr.Repr(Text(n!))); break;
            case Kind.List:
            {
                sb.Append('[');
                var first = true;
                foreach (var item in (JsonArray)n!)
                {
                    if (!first) sb.Append(", ");
                    first = false;
                    WriteRepr(item, sb);
                }
                sb.Append(']');
                break;
            }
            default:
            {
                sb.Append('{');
                var first = true;
                foreach (var (k, v) in (JsonObject)n!)
                {
                    if (!first) sb.Append(", ");
                    first = false;
                    sb.Append(PyStr.Repr(k)).Append(": ");
                    WriteRepr(v, sb);
                }
                sb.Append('}');
                break;
            }
        }
    }

    /// <summary>A JSON integer for a Python int.</summary>
    public static JsonNode Int(BigInteger i) =>
        i >= long.MinValue && i <= long.MaxValue ? JsonValue.Create((long)i) : JsonNode.Parse(i.ToString(CultureInfo.InvariantCulture))!;
}
