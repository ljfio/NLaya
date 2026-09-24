namespace NLaya.Extensions.AI;

/// <summary>Which confidence <see cref="LayaRouterChatClientOptions.ConfidenceThreshold"/> is compared against.</summary>
public enum RouteConfidence
{
    /// <summary>
    /// <see cref="Answer.Confidence"/>, the normalized-entropy value Python's <c>LayaRouter</c> gates on.
    /// It is not calibrated.
    /// </summary>
    Entropy,

    /// <summary><see cref="Answer.AnswerConfidence"/>: the calibrated probability of the chosen route.</summary>
    Answer,
}
