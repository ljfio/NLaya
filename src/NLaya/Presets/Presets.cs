using System.Text.Json.Nodes;

namespace NLaya;

/// <summary>Ready-made question sets from <c>laya.presets</c> (embedded verbatim from the Python library).</summary>
public static class Presets
{
    private static readonly JsonNode Data = Resources.Json("Presets.presets.json");

    private static Questions Get(string name) => Questions.FromJson(Data[name]!.DeepClone());

    /// <summary>Customer-support ticket triage: intent, urgency, frustration, refund, churn.</summary>
    public static Questions Triage() => Get("triage_questions");

    /// <summary>Inbound email routing and threat filtering; <paramref name="categories"/> replaces the team list.</summary>
    public static Questions Email(IEnumerable<KeyValuePair<string, string?>>? categories = null)
    {
        var q = Data["email_questions"]!.DeepClone();
        if (categories is not null)
        {
            var c = new JsonObject();
            foreach (var (k, v) in categories) c[k] = v;
            q["category"]!["criteria"] = c;
        }
        return Questions.FromJson(q);
    }

    /// <summary>Guardrails for an LLM app: jailbreak, prompt injection, sensitive data, harm severity, topic.</summary>
    public static Questions Guard() => Get("guard_questions");

    /// <summary>Content moderation: toxicity, harassment, threats, spam, severity.</summary>
    public static Questions Moderation() => Get("moderation_questions");

    /// <summary>Request routing signals: difficulty, domain, whether tools are needed, sensitivity.</summary>
    public static Questions Router() => Get("router_questions");
}
