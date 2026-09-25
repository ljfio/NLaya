using Microsoft.Extensions.AI;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Options;

using NLaya.Extensions.AI;
using NLaya.Hub;
using NLaya.Routing;

namespace NLaya.Tests.ExtensionsAI;

public class DependencyInjectionTests
{
    /// <summary>The fake backend still needs a real checkpoint's config and tokenizer.</summary>
    private static void RequireCachedCheckpoint()
    {
        if (HfCache.Snapshot(Laya.MultilingualModel) is null)
            Assert.Skip($"{Laya.MultilingualModel} is not cached; run: {HfCache.DownloadCommand(Laya.MultilingualModel)}");
    }

    private static ServiceProvider Provider(Action<IServiceCollection> register, Dictionary<string, string?>? config = null)
    {
        var services = new ServiceCollection();
        services.AddSingleton<IConfiguration>(new ConfigurationBuilder().AddInMemoryCollection(config ?? []).Build());
        register(services);
        return services.BuildServiceProvider(new ServiceProviderOptions { ValidateOnBuild = true, ValidateScopes = true });
    }

    [Fact]
    public async Task AddLaya_registers_one_agent_as_the_predictor_and_the_container_disposes_it()
    {
        RequireCachedCheckpoint();
        var factory = new FakeBackendFactory();
        FakeBackendFactory.FakeBackend backend;
        await using (var sp = Provider(s => s.AddLaya(Laya.MultilingualModel, o => o.Backend = factory)))
        {
            var agent = sp.GetRequiredService<LayaAgent>();
            Assert.Same(agent, sp.GetRequiredService<LayaAgent>());
            Assert.Same(agent, sp.GetRequiredService<ILayaPredictor>());
            Assert.Equal(Laya.MultilingualModel, agent.ModelId);
            Assert.Equal("fake", agent.Backend.Name);
            Assert.NotNull(agent.Predict("hello", Presets.Guard()).Noul("jailbreak"));
            backend = Assert.Single(factory.Backends);
        }
        Assert.True(backend.Disposed);
    }

    [Fact]
    public void Keyed_agents_resolve_separately()
    {
        RequireCachedCheckpoint();
        var factory = new FakeBackendFactory();
        using var sp = Provider(s => s
            .AddKeyedLaya("a", Laya.MultilingualModel, o => o.Backend = factory)
            .AddKeyedLaya("b", Laya.MultilingualModel, o => o.Backend = factory));
        var a = sp.GetRequiredKeyedService<LayaAgent>("a");
        Assert.NotSame(a, sp.GetRequiredKeyedService<LayaAgent>("b"));
        Assert.Same(a, sp.GetRequiredKeyedService<ILayaPredictor>("a"));
        Assert.Null(sp.GetService<ILayaPredictor>());
    }

    [Fact]
    public void Settings_bind_from_configuration_and_code_wins()
    {
        RequireCachedCheckpoint();
        var factory = new FakeBackendFactory();
        using var sp = Provider(s => s.AddLaya(o => o.Backend = factory), new()
        {
            ["Laya:Model"] = Laya.MultilingualModel,
            ["Laya:Revision"] = "main",
            ["Laya:Warmup"] = "false",
            ["Laya:Preload:0"] = "english",
            ["Laya:english:Model"] = "someone/else",
        });
        var settings = sp.GetRequiredService<IOptionsMonitor<LayaSettings>>();
        Assert.Equal((Laya.MultilingualModel, "main", false),
            (settings.CurrentValue.Model, settings.CurrentValue.Revision, settings.CurrentValue.Warmup));
        Assert.Equal(["english"], settings.CurrentValue.Preload!);
        Assert.Equal(Laya.MultilingualModel, sp.GetRequiredService<LayaAgent>().ModelId);

        using var coded = Provider(s => s.AddLaya(Laya.MultilingualModel, o => o.Backend = factory), new() { ["Laya:Model"] = "someone/else" });
        Assert.Equal(Laya.MultilingualModel, coded.GetRequiredService<LayaAgent>().ModelId);
    }

    [Fact]
    public async Task Warmup_loads_at_host_start_unless_disabled()
    {
        RequireCachedCheckpoint();
        foreach (var warmup in new[] { true, false })
        {
            var factory = new FakeBackendFactory();
            await using var sp = Provider(s => s.AddLaya(Laya.MultilingualModel, o => o.Backend = factory),
                new() { ["Laya:Warmup"] = warmup.ToString() });
            foreach (var service in sp.GetServices<IHostedService>()) await service.StartAsync(TestContext.Current.CancellationToken);
            Assert.Equal(warmup ? 1 : 0, factory.Created);
        }
    }

    [Fact]
    public void AddLayaRouter_registers_the_router_as_the_predictor()
    {
        var loads = new List<Checkpoint>();
        using var sp = Provider(s => s.AddLayaRouter(o => o.ConfigureAgent = (name, _) => loads.Add(name)),
            new() { ["Laya:Default"] = "ml" });
        var router = sp.GetRequiredService<Router>();
        Assert.Same(router, sp.GetRequiredService<ILayaPredictor>());
        Assert.Equal(Checkpoint.Multilingual, router.Default); // aliases work in configuration
        Assert.Empty(loads); // nothing loads until first use (or warm-up)
    }

    [Fact]
    public async Task UseLayaGuardrail_resolves_the_registered_predictor()
    {
        var predictor = new FakePredictor((_, _) => new() { ["jailbreak"] = FakePredictor.Noul(0.9) });
        using var sp = Provider(s => s
            .AddSingleton<ILayaPredictor>(predictor)
            .AddChatClient(new EchoChatClient()).UseLayaGuardrail(o => o.Action = GuardrailAction.Filter));
        var response = await sp.GetRequiredService<IChatClient>().GetResponseAsync("x", cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal(ChatFinishReason.ContentFilter, response.FinishReason);
        Assert.Single(predictor.Calls);
    }

    [Fact]
    public void Registered_hooks_run_on_agents_and_routers_and_can_take_dependencies()
    {
        RequireCachedCheckpoint();
        var log = new HookLog();
        using var sp = Provider(s => s
            .AddSingleton(log)
            .AddLayaHook<LoggingHook>()
            .AddLaya(Laya.MultilingualModel, o => o.Backend = new FakeBackendFactory())
            .AddLayaRouter());

        var agent = sp.GetRequiredService<LayaAgent>();
        var result = agent.Predict("hello", Presets.Guard());
        Assert.Contains(sp.GetRequiredService<Router>().Hooks, h => h is LoggingHook);
        Assert.Equal([Laya.MultilingualModel], log.Ends);
        Assert.Equal(Laya.MultilingualModel, result.Model);
        Assert.Equal(LayaResult.PythonModelName, result.ToJson()["model"]!.GetValue<string>());
    }

    internal sealed class HookLog
    {
        public List<string?> Ends { get; } = [];
    }

    internal sealed class LoggingHook(HookLog log) : ILayaHook
    {
        public void OnPredictEnd(PredictContext ctx)
        {
            Assert.NotNull(ctx.Elapsed);
            log.Ends.Add(ctx.Model);
        }
    }
}
