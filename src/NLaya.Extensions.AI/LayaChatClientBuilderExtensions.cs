using Microsoft.Extensions.DependencyInjection;

using NLaya;
using NLaya.Extensions.AI;

namespace Microsoft.Extensions.AI;

/// <summary>Adds NLaya middleware to a <see cref="ChatClientBuilder"/> pipeline.</summary>
public static class LayaChatClientBuilderExtensions
{
    /// <summary>
    /// Screen each request with <paramref name="predictor"/> (a <see cref="LayaAgent"/> or Router) before
    /// the inner client sees it. See <see cref="LayaGuardrailChatClient"/>.
    /// </summary>
    public static ChatClientBuilder UseLayaGuardrail(this ChatClientBuilder builder, ILayaPredictor predictor,
        Action<LayaGuardrailOptions>? configure = null)
    {
        ArgumentNullException.ThrowIfNull(predictor);
        var options = Options(configure);
        return builder.Use(inner => new LayaGuardrailChatClient(inner, predictor, options));
    }

    /// <summary>
    /// Screen each request with the <see cref="ILayaPredictor"/> registered in the container
    /// (<c>AddLaya</c> or <c>AddLayaRouter</c>), or the keyed one for <paramref name="serviceKey"/>.
    /// </summary>
    public static ChatClientBuilder UseLayaGuardrail(this ChatClientBuilder builder, Action<LayaGuardrailOptions>? configure = null,
        object? serviceKey = null)
    {
        var options = Options(configure);
        return builder.Use((inner, sp) => new LayaGuardrailChatClient(inner,
            serviceKey is null ? sp.GetRequiredService<ILayaPredictor>() : sp.GetRequiredKeyedService<ILayaPredictor>(serviceKey),
            options));
    }

    private static LayaGuardrailOptions Options(Action<LayaGuardrailOptions>? configure)
    {
        var options = new LayaGuardrailOptions();
        configure?.Invoke(options);
        return options;
    }
}
