using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;

using Xunit;

namespace NLaya.Testing;

/// <summary>Compares JSON as Python fixtures record it: same keys in the same order, numbers within a tolerance.</summary>
public static class JsonAssert
{
    /// <summary>Structural equality with numbers compared within <paramref name="tol"/>.</summary>
    public static void Equivalent(JsonNode? expected, JsonNode? actual, double tol = 1e-9) =>
        Equivalent(expected, actual, _ => tol);

    /// <summary>Structural equality with each number compared within <paramref name="tol"/> of its JSON path (<c>$.a[0].b</c>).</summary>
    public static void Equivalent(JsonNode? expected, JsonNode? actual, Func<string, double> tol) =>
        Equivalent(expected, actual, tol, "$");

    private static void Equivalent(JsonNode? expected, JsonNode? actual, Func<string, double> tol, string path)
    {
        switch (expected)
        {
            case null:
                Assert.True(actual is null, $"{path}: expected null, got {actual?.ToJsonString()}");
                break;
            case JsonObject eo:
                var ao = Assert.IsType<JsonObject>(actual);
                Assert.True(eo.Select(k => k.Key).SequenceEqual(ao.Select(k => k.Key)),
                    $"{path}: keys [{string.Join(",", eo.Select(k => k.Key))}] vs [{string.Join(",", ao.Select(k => k.Key))}]");
                foreach (var (k, v) in eo) Equivalent(v, ao[k], tol, $"{path}.{k}");
                break;
            case JsonArray ea:
                var aa = Assert.IsType<JsonArray>(actual);
                Assert.True(ea.Count == aa.Count, $"{path}: {ea.Count} items expected, got {aa.Count}");
                for (var i = 0; i < ea.Count; i++) Equivalent(ea[i], aa[i], tol, $"{path}[{i}]");
                break;
            case JsonValue ev when ev.GetValueKind() == JsonValueKind.Number:
                var e = Number(ev);
                var a = Number(actual ?? throw new Xunit.Sdk.XunitException($"{path}: expected {e}, got null"));
                Assert.True(Math.Abs(e - a) <= tol(path), $"{path}: expected {e}, got {a}");
                break;
            default:
                Assert.True(JsonNode.DeepEquals(expected, actual), $"{path}: expected {expected.ToJsonString()}, got {actual?.ToJsonString()}");
                break;
        }
    }

    private static double Number(JsonNode n) => double.Parse(n.ToJsonString(), CultureInfo.InvariantCulture);
}
