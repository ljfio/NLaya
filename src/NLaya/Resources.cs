using System.Text.Json.Nodes;

namespace NLaya;

/// <summary>JSON tables embedded from the Python library (regenerate with tools/fixtures/make_fixtures.py).</summary>
internal static class Resources
{
    public static JsonNode Json(string name)
    {
        using var s = typeof(Resources).Assembly.GetManifestResourceStream("NLaya." + name)
            ?? throw new InvalidOperationException($"missing embedded resource {name}");
        return JsonNode.Parse(s)!;
    }
}
