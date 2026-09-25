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

    private static Router RouterOver(ILayaBackendFactory factory) =>
        new(new RouterOptions { StandaloneRepos = true, ConfigureAgent = (_, o) => o.Backend = factory });

    [Fact]
    public async Task A_slow_load_does_not_block_a_loaded_checkpoint_and_is_shared()
    {
        RequireCachedCheckpoints();
        var factory = new GatedFactory("multilingual");
        using var router = RouterOver(factory);
        var english = router.Load("english");

        var first = Task.Run(() => router.Load("multilingual"), TestContext.Current.CancellationToken);
        Assert.True(factory.Entered.Wait(TimeSpan.FromSeconds(30), TestContext.Current.CancellationToken));
        var second = Task.Run(() => router.Load("multilingual"), TestContext.Current.CancellationToken);

        // The multilingual load is parked inside the factory; english is still served, and not yet listed as loaded.
        Assert.Same(english, router.Load("english"));
        Assert.Equal(["english"], router.Loaded);

        factory.Release.Set();
        var agents = await Task.WhenAll(first, second);
        Assert.Same(agents[0], agents[1]);
        Assert.Equal(1, factory.GatedCreates);
        Assert.Equal(["english", "multilingual"], router.Loaded);
    }

    [Fact]
    public void A_failed_load_is_retried()
    {
        RequireCachedCheckpoints();
        var factory = new FailOnceFactory();
        using var router = RouterOver(factory);

        Assert.Throws<IOException>(() => router.Load("english"));
        Assert.Empty(router.Loaded);
        Assert.NotNull(router.Load("english"));
        Assert.Equal(["english"], router.Loaded);
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
