using Microsoft.Extensions.AI;

namespace NLaya.Extensions.AI;

/// <summary>Settings for <see cref="LayaGuardrailChatClient"/>, with the defaults of Python's <c>LayaGuardrail</c>.</summary>
public sealed class LayaGuardrailOptions
{
    /// <summary>The questions to screen with; <see cref="Presets.Guard"/> when null. Choice questions never count as violations.</summary>
    public Questions? Questions { get; set; }

    /// <summary>What to do on a violation: throw (<see cref="GuardrailAction.Raise"/>, the default), answer with <see cref="RejectionMessage"/>, or annotate and let it through.</summary>
    public GuardrailAction Action { get; set; } = GuardrailAction.Raise;

    /// <summary>The assistant reply for <see cref="GuardrailAction.Filter"/>.</summary>
    public string RejectionMessage { get; set; } = "I cannot fulfill this request because it violates safety guidelines.";

    /// <summary>
    /// A noul answer is a violation when P(true) &gt;= this, and a score answer when its expected level
    /// is &gt;= this. Python applies one threshold to both, so for score questions the default 0.5 means
    /// "expected level 0.5 or more". With <see cref="Presets.Guard"/> that flags ordinary requests
    /// (<c>harm_severity</c> is about 0.5 for "When does the office open?"), so set a per-question
    /// threshold such as <c>Thresholds["harm_severity"] = 2</c> ("serious").
    /// </summary>
    public double Threshold { get; set; } = 0.5;

    /// <summary>Per-question thresholds, by question id, overriding <see cref="Threshold"/>.</summary>
    public IDictionary<string, double> Thresholds { get; } = new Dictionary<string, double>(StringComparer.Ordinal);

    /// <summary>
    /// The text to screen (Python's <c>state_key</c>). Defaults to the newest user message. When it
    /// returns null or whitespace the request is not screened.
    /// </summary>
    public Func<IEnumerable<ChatMessage>, string?>? SelectText { get; set; }

    /// <summary>Per-call prediction settings, e.g. <see cref="Routing.RouteOptions"/> when the predictor is a Router.</summary>
    public PredictOptions? PredictOptions { get; set; }
}
