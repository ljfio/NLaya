using System.Text.Json.Nodes;

namespace NLaya;

/// <summary>Token accounting. Laya never generates, so <see cref="OutputTokens"/> is always 0.</summary>
public sealed record Usage(int InputTokens, int OutputTokens = 0)
{
    public static readonly Usage Zero = new(0, 0);

    public static Usage Sum(IEnumerable<LayaResult> results)
    {
        int i = 0, o = 0;
        foreach (var r in results)
        {
            i += r.Usage.InputTokens;
            o += r.Usage.OutputTokens;
        }
        return new Usage(i, o);
    }

    public JsonObject ToJson() => new() { ["input_tokens"] = InputTokens, ["output_tokens"] = OutputTokens };
}
