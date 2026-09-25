using NLaya.Backends;
using NLaya.Hub;
using NLaya.Routing;
using NLaya.Tests.ExtensionsAI;

namespace NLaya.Tests;

/// <summary>Router checkpoint loads run outside its lock, are shared by concurrent callers, and can be retried.</summary>
public class RouterLoadingTests
{
    private static void RequireCachedCheckpoints()
    {
        foreach (var repo in new[] { Laya.DefaultModel, Laya.MultilingualModel })
            if (HfCache.Snapshot(repo) is null)
                Assert.Skip($"{repo} is not cached; run: {HfCache.DownloadCommand(repo)}");
    }

    private static Router RouterOver(ILayaBackendFactory factory, int maxLoaded = 2) =>
        new(new RouterOptions { StandaloneRepos = true, MaxLoaded = maxLoaded, ConfigureAgent = (_, o) => o.Backend = factory });

    private static bool IsDisposed(LayaAgent agent)
    {
        try { _ = agent.Backend; return false; }
        catch (ObjectDisposedException) { return true; }
    }

    [Fact]
    public async Task A_slow_load_does_not_block_a_loaded_checkpoint_and_is_shared()
    {
        RequireCachedCheckpoints();
        var factory = new GatedFactory("multilingual");
        using var router = RouterOver(factory);
        var english = router.Load(Checkpoint.English);

        var first = Task.Run(() => router.Load(Checkpoint.Multilingual), TestContext.Current.CancellationToken);
        Assert.True(factory.Entered.Wait(TimeSpan.FromSeconds(30), TestContext.Current.CancellationToken));
        var second = Task.Run(() => router.Load(Checkpoint.Multilingual), TestContext.Current.CancellationToken);

        // The multilingual load is parked inside the factory; english is still served, and not yet listed as loaded.
        Assert.Same(english, router.Load(Checkpoint.English));
        Assert.Equal([Checkpoint.English], router.Loaded);

        factory.Release.Set();
        var agents = await Task.WhenAll(first, second);
        Assert.Same(agents[0], agents[1]);
        Assert.Equal(1, factory.GatedCreates);
        Assert.Equal([Checkpoint.English, Checkpoint.Multilingual], router.Loaded);
    }

    [Fact]
    public void A_failed_load_is_retried()
    {
        RequireCachedCheckpoints();
        var factory = new FailOnceFactory();
        using var router = RouterOver(factory);

        Assert.Throws<IOException>(() => router.Load(Checkpoint.English));
        Assert.Empty(router.Loaded);
        Assert.NotNull(router.Load(Checkpoint.English));
        Assert.Equal([Checkpoint.English], router.Loaded);
    }

    [Fact]
    public void An_idle_evicted_agent_is_disposed_at_once()
    {
        RequireCachedCheckpoints();
        using var router = RouterOver(new FakeBackendFactory(), maxLoaded: 1);
        var english = router.Load(Checkpoint.English);
        router.Load(Checkpoint.Multilingual);
        Assert.True(IsDisposed(english));
        Assert.Equal([Checkpoint.Multilingual], router.Loaded);
    }

    [Fact]
    public async Task An_agent_evicted_mid_call_is_disposed_when_the_call_returns()
    {
        RequireCachedCheckpoints();
        var factory = new BlockingRunFactory();
        using var router = RouterOver(factory, maxLoaded: 1);
        var english = router.Load(Checkpoint.English);
        var call = Task.Run(() => router.Predict("hello", Presets.Triage(), new RouteOptions { Checkpoint = Checkpoint.English }),
            TestContext.Current.CancellationToken);
        Assert.True(factory.Entered.Wait(TimeSpan.FromSeconds(30), TestContext.Current.CancellationToken));

        router.Load(Checkpoint.Multilingual);   // evicts english while its call is running
        Assert.False(IsDisposed(english));

        factory.Release.Set();
        await call;
        Assert.True(IsDisposed(english));
    }

    [Fact]
    public void Unloading_an_attached_agent_lets_the_router_own_the_next_load()
    {
        RequireCachedCheckpoints();
        var factory = new FakeBackendFactory();
        var router = RouterOver(factory);
        using var mine = Laya.Load(Laya.DefaultModel, o => o.Backend = factory);
        router.Attach(Checkpoint.English, mine);
        router.Unload(Checkpoint.English);
        var loaded = router.Load(Checkpoint.English);
        Assert.NotSame(mine, loaded);

        router.Dispose();
        Assert.True(IsDisposed(loaded));
        Assert.False(IsDisposed(mine));
    }

    /// <summary>Backends whose first <c>Run</c> waits for <see cref="Release"/>.</summary>
    private sealed class BlockingRunFactory : ILayaBackendFactory
    {
        private readonly FakeBackendFactory _inner = new();
        public ManualResetEventSlim Entered { get; } = new();
        public ManualResetEventSlim Release { get; } = new();

        public IReadOnlyList<string> RequiredFiles => [];

        public ILayaBackend Create(LayaCheckpoint checkpoint) => new Blocking(_inner.Create(checkpoint), this);

        private sealed class Blocking(ILayaBackend inner, BlockingRunFactory owner) : ILayaBackend
        {
            public string Name => inner.Name;

            public BackendOutput Run(EncodedBatch batch)
            {
                owner.Entered.Set();
                owner.Release.Wait(TimeSpan.FromSeconds(30));
                return inner.Run(batch);
            }

            public void Dispose() => inner.Dispose();
        }
    }

    /// <summary>Blocks backend creation for checkpoints whose id contains <c>gated</c> until <see cref="Release"/> is set.</summary>
    private sealed class GatedFactory(string gated) : ILayaBackendFactory
    {
        private readonly FakeBackendFactory _inner = new();
        public ManualResetEventSlim Entered { get; } = new();
        public ManualResetEventSlim Release { get; } = new();
        public int GatedCreates;

        public IReadOnlyList<string> RequiredFiles => [];

        public ILayaBackend Create(LayaCheckpoint checkpoint)
        {
            if (checkpoint.ModelId.Contains(gated, StringComparison.Ordinal))
            {
                Interlocked.Increment(ref GatedCreates);
                Entered.Set();
                Release.Wait(TimeSpan.FromSeconds(30));
            }
            return _inner.Create(checkpoint);
        }
    }

    private sealed class FailOnceFactory : ILayaBackendFactory
    {
        private readonly FakeBackendFactory _inner = new();
        private int _calls;

        public IReadOnlyList<string> RequiredFiles => [];

        public ILayaBackend Create(LayaCheckpoint checkpoint) =>
            Interlocked.Increment(ref _calls) == 1 ? throw new IOException("disk hiccup") : _inner.Create(checkpoint);
    }
}
