using NLaya.Hub;
using NLaya.Lang;
using NLaya.Routing;
using NLaya.Tests.ExtensionsAI;

namespace NLaya.Tests;

/// <summary>Public members with no Python fixture of their own: aliases, hook registries and router lifecycle.</summary>
public class PublicApiTests
{
    private static LayaAgent FakeAgent()
    {
        if (HfCache.Snapshot(Laya.MultilingualModel) is null)
            Assert.Skip($"{Laya.MultilingualModel} is not cached; run: {HfCache.DownloadCommand(Laya.MultilingualModel)}");
        return Laya.Load(Laya.MultilingualModel, o => o.Backend = new FakeBackendFactory());
    }

    [Fact]
    public void Choice_without_options_binds_to_labels_and_throws()
    {
        Assert.Throws<ArgumentException>(() => Question.Choice("Which?"));
        Assert.Throws<ArgumentException>(() => new Questions().Choice("x", "Which?"));
        Assert.Equal(["a", "b"], Question.Choice("Which?", new List<string> { "a", "b" }).Options.Select(o => o.Key));
    }

    [Fact]
    public void Script_and_Latin_language_helpers()
    {
        Assert.Equal("latin", LanguageDetector.DetectScript("The invoice is wrong"));
        Assert.Equal("han", LanguageDetector.DetectScript("你好世界"));
        Assert.Equal("unknown", LanguageDetector.DetectScript("1234 !!"));
        Assert.Equal("es", LanguageDetector.GuessLatinLanguage("Hola, quiero cancelar mi pedido porque no llegó a tiempo y es para mañana"));
    }

    [Fact]
    public void MostLikelyLevel_is_the_argmax()
    {
        var answer = new ScoreAnswer
        {
            Score = 1.2,
            Legend = [],
            Probabilities = [0.1, 0.6, 0.3],
            Confidence = 0.5,
            AnswerConfidence = 0.5,
            Action = new ActionInfo(0.5),
        };
        Assert.Equal(1, answer.MostLikelyLevel);
    }

    [Fact]
    public void Each_context_gets_its_own_run_id()
    {
        var a = new PredictContext([], new Questions());
        var b = new PredictContext([], new Questions());
        Assert.Matches("^[0-9a-f]{32}$", a.RunId);
        Assert.NotEqual(a.RunId, b.RunId);
    }

    [Fact]
    public void Hooks_can_be_removed_and_process_defaults_managed()
    {
        using var agent = FakeAgent();
        var hook = LayaHooks.OnStart(_ => { });
        agent.AddHook(hook);
        Assert.True(agent.RemoveHook(hook));
        Assert.False(agent.RemoveHook(hook));
        Assert.Empty(agent.Hooks);

        // Process-wide state: only ever add a no-op hook here, and restore what was there.
        var before = LayaHooks.Defaults.ToArray();
        try
        {
            LayaHooks.SetDefaults(hook);
            LayaHooks.AddDefault(hook);
            Assert.Equal([hook, hook], LayaHooks.Defaults);
            LayaHooks.ClearDefaults();
            Assert.Empty(LayaHooks.Defaults);
        }
        finally
        {
            LayaHooks.SetDefaults(before);
        }
    }

    [Fact]
    public async Task Router_attach_preload_and_unload()
    {
        using var attached = FakeAgent();
        var factory = new FakeBackendFactory();
        using var router = new Router(new RouterOptions { StandaloneRepos = true, ConfigureAgent = (_, o) => o.Backend = factory });

        Assert.Same(attached, router.Attach(Checkpoint.Multilingual, attached));
        Assert.Equal([Checkpoint.Multilingual], router.Loaded);

        if (HfCache.Snapshot(Laya.DefaultModel) is null)
            Assert.Skip($"{Laya.DefaultModel} is not cached; run: {HfCache.DownloadCommand(Laya.DefaultModel)}");
        await router.PreloadAsync([Checkpoint.English], TestContext.Current.CancellationToken);
        Assert.Equal([Checkpoint.English, Checkpoint.Multilingual], router.Loaded.Order());
        Assert.Equal(1, factory.Created);

        router.Unload(Checkpoint.English);
        Assert.Equal([Checkpoint.Multilingual], router.Loaded);
        router.Unload();
        Assert.Empty(router.Loaded);
    }
}
