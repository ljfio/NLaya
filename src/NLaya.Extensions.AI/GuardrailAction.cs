namespace NLaya.Extensions.AI;

/// <summary>What <see cref="LayaGuardrailChatClient"/> does when a request fails the guardrail (Python's <c>action</c>).</summary>
public enum GuardrailAction
{
    /// <summary>Throw <see cref="LayaGuardrailException"/>; the inner client is not called.</summary>
    Raise,

    /// <summary>Answer with <see cref="LayaGuardrailOptions.RejectionMessage"/>; the inner client is not called.</summary>
    Filter,

    /// <summary>Call the inner client anyway and attach the <see cref="GuardrailResult"/> to the response.</summary>
    Annotate,
}
