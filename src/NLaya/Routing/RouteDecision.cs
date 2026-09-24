using System.Text.Json.Nodes;

using NLaya.Lang;

namespace NLaya.Routing;

/// <summary>Which checkpoint a request goes to, why, and what detection saw (Python <c>RouteDecision</c>).</summary>
public sealed record RouteDecision(string Model, string Repo, string Reason, LanguageDetection? Detection = null, string? Workflow = null)
{
    public JsonObject ToJson() => new()
    {
        ["model"] = Model,
        ["repo"] = Repo,
        ["reason"] = Reason,
        ["detection"] = Detection?.ToJson(),
        ["workflow"] = Workflow,
    };
}
