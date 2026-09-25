using System.Diagnostics.CodeAnalysis;

using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

using NLaya;
using NLaya.Extensions.AI;
using NLaya.Routing;

namespace Microsoft.Extensions.DependencyInjection;

/// <summary>
/// Registers NLaya with a service container. Agents and Routers are thread-safe and expensive to
/// build, so they are singletons, disposed with the container and (by default) loaded at host start.
/// </summary>
public static class LayaServiceCollectionExtensions
{
    /// <summary>The named <see cref="LayaSettings"/> a Router reads; bound from the <c>"Laya"</c> section.</summary>
    public const string RouterSettingsName = "NLaya.Router";

    /// <summary>
    /// A singleton <see cref="LayaAgent"/>, also resolvable as <see cref="ILayaPredictor"/> (unless one is
    /// already registered). The model is <paramref name="model"/>, else <c>Laya:Model</c>, else
    /// <see cref="Laya.DefaultModel"/>.
    /// <code>services.AddLaya(Laya.MultilingualModel, o => o.UseTorchSharp());</code>
    /// </summary>
    public static IServiceCollection AddLaya(this IServiceCollection services, string? model, Action<LayaOptions> configure)
    {
        ArgumentNullException.ThrowIfNull(configure);
        BindSettings(services, Options.Options.DefaultName, LayaSettings.SectionName);
        services.TryAddSingleton(sp => CreateAgent(sp, Options.Options.DefaultName, model, configure));
        services.TryAddSingleton<ILayaPredictor>(sp => sp.GetRequiredService<LayaAgent>());
        AddWarmup(services, Options.Options.DefaultName, sp => sp.GetRequiredService<LayaAgent>());
        return services;
    }

    /// <inheritdoc cref="AddLaya(IServiceCollection, string?, Action{LayaOptions})"/>
    public static IServiceCollection AddLaya(this IServiceCollection services, Action<LayaOptions> configure) =>
        services.AddLaya(null, configure);

    /// <summary>
    /// A keyed singleton <see cref="LayaAgent"/> (and keyed <see cref="ILayaPredictor"/>), for apps that use
    /// several checkpoints. Settings bind from <c>"Laya:&lt;key&gt;"</c>.
    /// <code>services.AddKeyedLaya("english", Laya.DefaultModel, o => o.UseTorchSharp());</code>
    /// </summary>
    public static IServiceCollection AddKeyedLaya(this IServiceCollection services, string key, string? model, Action<LayaOptions> configure)
    {
        ArgumentException.ThrowIfNullOrEmpty(key);
        ArgumentNullException.ThrowIfNull(configure);
        BindSettings(services, key, $"{LayaSettings.SectionName}:{key}");
        services.TryAddKeyedSingleton(key, (sp, _) => CreateAgent(sp, key, model, configure));
        services.TryAddKeyedSingleton<ILayaPredictor>(key, (sp, k) => sp.GetRequiredKeyedService<LayaAgent>(k));
        AddWarmup(services, key, sp => sp.GetRequiredKeyedService<LayaAgent>(key));
        return services;
    }

    /// <summary>
    /// A singleton <see cref="Router"/>, also resolvable as <see cref="ILayaPredictor"/> (unless one is
    /// already registered). Router settings bind from the <c>"Laya"</c> section; warm-up preloads
    /// <see cref="LayaSettings.Preload"/>, or the router's default checkpoint.
    /// <code>services.AddLayaRouter(o => o.ConfigureAgent = (_, a) => a.UseTorchSharp());</code>
    /// </summary>
    public static IServiceCollection AddLayaRouter(this IServiceCollection services, Action<RouterOptions>? configure = null)
    {
        BindSettings(services, RouterSettingsName, LayaSettings.SectionName);
        services.TryAddSingleton(sp => CreateRouter(sp, configure));
        services.TryAddSingleton<ILayaPredictor>(sp => sp.GetRequiredService<Router>());
        AddWarmup(services, RouterSettingsName, sp =>
        {
            var router = sp.GetRequiredService<Router>();
            router.Preload(Settings(sp, RouterSettingsName).Preload?.Select(Checkpoints.Parse) ?? [router.Default]);
        });
        return services;
    }

    /// <summary>
    /// A hook for every agent and router this container builds, created by the container (so it can take
    /// dependencies). The .NET alternative to <see cref="LayaHooks.SetDefaults"/>'s process-wide list.
    /// A router runs the hooks per request; the agents it loads don't run them again.
    /// </summary>
    public static IServiceCollection AddLayaHook<[DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicConstructors)] THook>(
        this IServiceCollection services) where THook : class, ILayaHook
    {
        services.TryAddEnumerable(ServiceDescriptor.Singleton<ILayaHook, THook>());
        return services;
    }

    /// <summary>An existing hook instance for every agent and router this container builds; see <see cref="AddLayaHook{THook}(IServiceCollection)"/>.</summary>
    public static IServiceCollection AddLayaHook(this IServiceCollection services, ILayaHook hook)
    {
        ArgumentNullException.ThrowIfNull(hook);
        services.AddSingleton(hook);
        return services;
    }

    private static void BindSettings(IServiceCollection services, string name, string section) =>
        services.AddOptions<LayaSettings>(name)
            .Configure<IServiceProvider>((s, sp) => sp.GetService<IConfiguration>()?.GetSection(section).Bind(s));

    private static LayaSettings Settings(IServiceProvider sp, string name) =>
        sp.GetRequiredService<IOptionsMonitor<LayaSettings>>().Get(name);

    private static LayaAgent CreateAgent(IServiceProvider sp, string settingsName, string? model, Action<LayaOptions> configure)
    {
        var s = Settings(sp, settingsName);
        return Laya.Load(model ?? s.Model ?? Laya.DefaultModel, o =>
        {
            // Before configure: backends copy the logger when they are added (UseTorchSharp).
            o.Logger = sp.GetService<ILoggerFactory>()?.CreateLogger(typeof(LayaAgent));
            ApplyTo(s, o);
            if (s.Subfolder is not null) o.Subfolder = s.Subfolder;
            foreach (var hook in sp.GetServices<ILayaHook>()) o.Hooks.Add(hook);
            configure(o);
        });
    }

    private static Router CreateRouter(IServiceProvider sp, Action<RouterOptions>? configure)
    {
        var s = Settings(sp, RouterSettingsName);
        var o = new RouterOptions { Logger = sp.GetService<ILoggerFactory>()?.CreateLogger(typeof(Router)) };
        if (s.MaxLoaded is { } maxLoaded) o.MaxLoaded = maxLoaded;
        if (s.Default is not null) o.Default = Checkpoints.Parse(s.Default);
        if (s.AutoTaskDetection is { } auto) o.AutoTaskDetection = auto;
        if (s.StandaloneRepos is { } standalone) o.StandaloneRepos = standalone;
        foreach (var hook in sp.GetServices<ILayaHook>()) o.Hooks.Add(hook);
        configure?.Invoke(o);
        var agent = o.ConfigureAgent;
        o.ConfigureAgent = (name, a) =>
        {
            ApplyTo(s, a);
            agent?.Invoke(name, a);
        };
        return new Router(o);
    }

    private static void ApplyTo(LayaSettings s, LayaOptions o)
    {
        if (s.Revision is not null) o.Revision = s.Revision;
        if (s.CacheDir is not null) o.CacheDir = s.CacheDir;
    }

    /// <summary>
    /// Added with <c>AddSingleton</c>, not <c>AddHostedService</c>: the latter de-duplicates by
    /// implementation type, which would keep only one warm-up per container.
    /// </summary>
    private static void AddWarmup(IServiceCollection services, string settingsName, Action<IServiceProvider> warmup) =>
        services.AddSingleton<IHostedService>(sp => new LayaWarmupService(() => Settings(sp, settingsName), () => warmup(sp)));
}
