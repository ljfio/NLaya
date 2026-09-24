namespace NLaya.Extensions.AI;

/// <summary>
/// Keys NLaya adds to <c>ChatResponse.AdditionalProperties</c> (and to the first streaming update).
/// Decisions travel on the response rather than on the client, because clients are shared across threads.
/// </summary>
public static class LayaChatProperties
{
    /// <summary>The <see cref="GuardrailResult"/> of the guardrail check.</summary>
    public const string Guardrail = "laya.guardrail";

    /// <summary>The route label <see cref="LayaRouterChatClient"/> sent the request to.</summary>
    public const string Route = "laya.route";

    /// <summary>The <see cref="LayaResult"/> behind the routing decision.</summary>
    public const string Result = "laya.result";
}
