using Microsoft.Extensions.AI;

namespace NLaya.Extensions.AI;

/// <summary>Settings for <see cref="LayaRouterChatClient"/>, with the defaults of Python's <c>LayaRouter</c>.</summary>
public sealed class LayaRouterChatClientOptions
{
    /// <summary>
    /// Route label -> destination, in order: the order is the order of the options Laya reads.
    /// <c>Routes = { ["simple"] = (small, "greetings, FAQs"), ["complex"] = (large, "reasoning, code") }</c>.
    /// </summary>
    public IDictionary<string, LayaRoute> Routes { get; } = new OrderedDictionary<string, LayaRoute>(StringComparer.Ordinal);

    /// <summary>The question Laya answers to choose a route.</summary>
    public string Instructions { get; set; } = "Which route should handle this request?";

    /// <summary>The route to use when confidence is below <see cref="ConfidenceThreshold"/>. Without one, the chosen route is kept.</summary>
    public string? Fallback { get; set; }

    /// <summary>Fall back when the route's confidence is below this. 0 (the default) never falls back.</summary>
    public double ConfidenceThreshold { get; set; }

    /// <summary>
    /// Which confidence the threshold applies to. <see cref="RouteConfidence.Entropy"/> matches Python;
    /// <see cref="RouteConfidence.Answer"/> is calibrated, so a threshold on it reads as a probability.
    /// </summary>
    public RouteConfidence ConfidenceMeasure { get; set; } = RouteConfidence.Entropy;

    /// <summary>The text to route on (Python's <c>state_key</c>). Defaults to the newest user message.</summary>
    public Func<IEnumerable<ChatMessage>, string?>? SelectText { get; set; }

    /// <summary>Per-call prediction settings, e.g. <see cref="Routing.RouteOptions"/> when the predictor is a Router.</summary>
    public PredictOptions? PredictOptions { get; set; }
}
