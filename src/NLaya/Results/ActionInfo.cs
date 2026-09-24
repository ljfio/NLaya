using System.Text.Json.Nodes;

namespace NLaya;

/// <summary>The act/escalate head's output for one question.</summary>
public sealed record ActionInfo(double ActProbability)
{
    public JsonObject ToJson() => new() { ["act_probability"] = ActProbability };
}
