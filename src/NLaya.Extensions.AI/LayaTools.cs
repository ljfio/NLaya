using Microsoft.Extensions.AI;

namespace NLaya.Extensions.AI;

/// <summary>Laya as <see cref="AIFunction"/> tools, so an LLM can call it inside an agent loop.</summary>
public static class LayaTools
{
    /// <summary>
    /// A tool taking <c>text</c> and returning Laya's answers to <paramref name="questions"/> as JSON
    /// (the shape of Python's <c>predict</c> result).
    /// </summary>
    public static AIFunction Create(ILayaPredictor predictor, Questions questions, string name, string description)
    {
        ArgumentNullException.ThrowIfNull(predictor);
        ArgumentNullException.ThrowIfNull(questions);
        return AIFunctionFactory.Create(
            async (string text, CancellationToken ct) => (await predictor.PredictAsync(text, questions, null, ct).ConfigureAwait(false)).ToJson(),
            name, description);
    }

    /// <summary>Support triage with <see cref="Presets.Triage"/>: intent, urgency, frustration, churn risk and refund request.</summary>
    public static AIFunction Triage(ILayaPredictor predictor) =>
        Create(predictor, Presets.Triage(), "laya_triage",
            "Triage a customer message: intent, urgency, frustration (0-3), churn risk and whether a refund is requested.");
}
