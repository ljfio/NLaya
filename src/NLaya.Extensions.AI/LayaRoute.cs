using Microsoft.Extensions.AI;

namespace NLaya.Extensions.AI;

/// <summary>A destination for <see cref="LayaRouterChatClient"/>: a client, and the description Laya reads to pick it.</summary>
public sealed record LayaRoute(IChatClient Client, string? Description = null)
{
    /// <summary>A route from a <c>(client, description)</c> tuple, for collection initializers.</summary>
    public static implicit operator LayaRoute((IChatClient Client, string? Description) route) => new(route.Client, route.Description);
}
