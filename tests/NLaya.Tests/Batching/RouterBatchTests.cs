using System.Text.Json.Nodes;

using NLaya.Calibration;
using NLaya.Hub;
using NLaya.Routing;
using NLaya.Tests.ExtensionsAI;

namespace NLaya.Tests.Batching;

public class RouterBatchTests
{
    private static readonly JsonNode Fixture = TestFiles.Fixture("router_batch.json");

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    private static List<RouteRequest> Requests() => Fixture["requests"]!.AsArray().Select(r =>
    {
        string? S(string k) => r![k]?.GetValue<string>();
        var guess = S("lang_guess");
        return new RouteRequest(LayaState.FromJson(r!["state"]!.DeepClone()), Questions.FromJson(r["questions"]))
        {
            Checkpoint = (S("model") ?? S("task")) is { } name ? Checkpoints.Parse(name) : null,
            Lang = S("lang"),
            LangGuess = guess is null ? null : _ => guess,
        };
    }).ToList();

    /// <summary>The fake backend still needs both standalone checkpoints' configs and tokenizers.</summary>
    private static void RequireCachedCheckpoints()
    {
        foreach (var repo in new[] { Laya.DefaultModel, Laya.MultilingualModel })
            if (HfCache.Snapshot(repo) is null)
                Assert.Skip($"{repo} is not cached; run: {HfCache.DownloadCommand(repo)}");
    }

    /// <summary>A router over fake agents, like the fixture's: multilingual has German temperatures.</summary>
    private static Router FakeRouter(FakeBackendFactory factory, List<(string Model, string[] States, string[] Ids, string? Lang)>? calls = null,
        int maxLoaded = 2, Action<LayaOptions>? configure = null) =>
        new(new RouterOptions
        {
            StandaloneRepos = true,
            MaxLoaded = maxLoaded,
            ConfigureAgent = (name, o) =>
            {
                o.Backend = factory;
                if (name == Checkpoint.Multilingual) o.LangTemperatures["de"] = new LanguageTemperature { Temperature = [1.0, 1.0, 1.0] };
                if (calls is not null)
                    o.Hooks.Add(LayaHooks.OnStart(c =>
                    {
                        lock (calls) calls.Add((name.Name(), c.States.Select(s => s.Serialize()).ToArray(), c.Questions.Keys.ToArray(), c.Lang));
                    }));
                configure?.Invoke(o);
            },
        });

    [Fact]
    public void RouteBatch_matches_python()
    {
        var expected = Fixture["decisions"]!.AsArray();
        var decisions = new Router().RouteBatch(Requests());

        Assert.Equal(expected.Count, decisions.Count);
        for (var i = 0; i < decisions.Count; i++)
        {
            Assert.Equal(expected[i]!["model"]!.GetValue<string>(), decisions[i].Checkpoint.Name());
            Assert.Equal(expected[i]!["reason"]!.GetValue<string>(), decisions[i].Reason);
        }
    }

    [Fact]
    public void RouteBatch_names_a_request_without_state()
    {
        var ex = Assert.Throws<ArgumentException>(() => new Router().RouteBatch([new RouteRequest("ok", new Questions()), new RouteRequest(null!, new Questions())]));
        Assert.StartsWith("request 1 is missing required key 'state'", ex.Message);
    }

    [Theory]
    [InlineData(2)]
    [InlineData(1)]
    public void PredictBatch_groups_like_python_and_loads_each_checkpoint_once(int maxLoaded)
    {
        RequireCachedCheckpoints();
        var factory = new FakeBackendFactory();
        var calls = new List<(string Model, string[] States, string[] Ids, string? Lang)>();
        using var router = FakeRouter(factory, calls, maxLoaded);

        var results = router.PredictBatch(Requests(), new BatchOptions { BatchSize = 4 });

        var expected = Fixture["calls"]!.AsArray();
        Assert.Equal(expected.Count, calls.Count);
        for (var i = 0; i < calls.Count; i++)
        {
            var e = expected[i]!;
            Assert.Equal(e["model"]!.GetValue<string>(), calls[i].Model);
            Assert.Equal(e["states"]!.AsArray().Select(s => LayaState.FromJson(s!.DeepClone()).Serialize()), calls[i].States);
            Assert.Equal(e["questions"]!.AsArray().Select(q => q!.GetValue<string>()), calls[i].Ids);
            Assert.Equal(e["lang"]?.GetValue<string>(), calls[i].Lang);
        }
        Assert.Equal(Fixture["result_models"]!.AsArray().Select(m => m!.GetValue<string>()), results.Select(r => r.Routing!.Checkpoint.Name()));
        Assert.Equal(2, factory.Created);
    }

    [Fact]
    public async Task Router_streams_requests_and_states_in_order()
    {
        RequireCachedCheckpoints();
        using var router = FakeRouter(new FakeBackendFactory());
        var requests = Requests();

        var streamed = await router.PredictStreamAsync(requests, new BatchOptions { BatchSize = 4 }, Ct).ToListAsync(Ct);
        Assert.Equal(Fixture["result_models"]!.AsArray().Select(m => m!.GetValue<string>()), streamed.Select(r => r.Routing!.Checkpoint.Name()));

        // Through ILayaPredictor: same questions for every state, each routed on its own.
        var states = requests.Where(r => r.Checkpoint is null && r.Lang is null && r.LangGuess is null).Select(r => r.State).ToList();
        var viaInterface = await ((ILayaPredictor)router).PredictStreamAsync(states, requests[0].Questions, new BatchOptions { BatchSize = 3 }, Ct).ToListAsync(Ct);
        Assert.Equal(states.Select(s => router.Route(s).Checkpoint), viaInterface.Select(r => r.Routing!.Checkpoint));
    }

    [Fact]
    public void Start_hooks_can_skip_or_rewrite_a_request_before_grouping()
    {
        RequireCachedCheckpoints();
        var calls = new List<(string Model, string[] States, string[] Ids, string? Lang)>();
        var cached = new LayaResult("cached", new OrderedDictionary<string, Answer>(), Usage.Zero);
        var hook = new RecordingHook(ctx =>
        {
            var text = ctx.States[0].Serialize();
            if (text == "skip me") ctx.Skip([cached]);
            if (text == "rewrite me") ctx.Questions = new Questions { ["other"] = Question.Noul("Something else?") };
        });
        using var router = FakeRouter(new FakeBackendFactory(), calls);
        router.AddHook(hook);
        var qs = new Questions { ["urgent"] = Question.Noul("Is it urgent?") };

        var results = router.PredictBatch([new RouteRequest("hello there, a normal request", qs), new("skip me", qs), new("rewrite me", qs)]);

        Assert.Equal("cached", results[1].Model);
        Assert.Equal(Checkpoint.English, results[1].Routing!.Checkpoint);
        Assert.Equal([["urgent"], ["other"]], calls.Select(c => c.Ids));
        Assert.Equal(["start:hello there, a normal request", "start:skip me", "start:rewrite me",
            "end:hello there, a normal request", "end:skip me", "end:rewrite me"], hook.Events);
    }

    [Fact]
    public void A_failing_group_errors_and_ends_every_started_request()
    {
        RequireCachedCheckpoints();
        var hook = new RecordingHook();
        using var router = FakeRouter(new FakeBackendFactory(),
            configure: o => o.Hooks.Add(LayaHooks.OnStart(_ => throw new InvalidOperationException("boom"))));
        router.AddHook(hook);
        var qs = new Questions { ["urgent"] = Question.Noul("Is it urgent?") };

        var ex = Assert.Throws<InvalidOperationException>(() => router.PredictBatch([new RouteRequest("first request", qs), new("second request", qs)]));

        Assert.Equal("boom", ex.Message);
        Assert.Equal(["start:first request", "start:second request", "error:first request", "error:second request",
            "end:first request", "end:second request"], hook.Events);
    }
}
