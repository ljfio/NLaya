using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace NLaya.Tests;

internal static class JsonAssert
{
    /// <summary>Structural equality with numbers compared within <paramref name="tol"/>.</summary>
    public static void Equivalent(JsonNode? expected, JsonNode? actual, double tol = 1e-9, string path = "$")
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
                Assert.Equal(ea.Count, aa.Count);
                for (var i = 0; i < ea.Count; i++) Equivalent(ea[i], aa[i], tol, $"{path}[{i}]");
                break;
            case JsonValue ev when ev.GetValueKind() == JsonValueKind.Number:
                var e = Number(ev);
                var a = Number(actual ?? throw new Xunit.Sdk.XunitException($"{path}: expected {e}, got null"));
                Assert.True(Math.Abs(e - a) <= tol, $"{path}: expected {e}, got {a}");
                break;
            default:
                Assert.True(JsonNode.DeepEquals(expected, actual), $"{path}: expected {expected.ToJsonString()}, got {actual?.ToJsonString()}");
                break;
        }
    }

    public static double Number(JsonNode n) => double.Parse(n.ToJsonString(), CultureInfo.InvariantCulture);
}
