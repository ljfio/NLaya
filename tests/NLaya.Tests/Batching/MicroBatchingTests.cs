using NLaya.Routing;

namespace NLaya.Tests.Batching;

public class MicroBatchingTests
{
    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    private static Questions Q(string instructions = "Is this urgent?") => new Questions().Noul("urgent", instructions);

    [Fact]
    public async Task Concurrent_requests_with_the_same_questions_share_a_batch()
    {
        var inner = new ChunkRecordingPredictor();
        await using var batcher = new MicroBatchingPredictor(inner, new MicroBatchOptions { MaxDelay = TimeSpan.FromSeconds(1), MaxBatchSize = 8 });

        // A fresh Questions per request, as a web handler would build them: equal content still batches.
        var results = await Task.WhenAll(Enumerable.Range(0, 8).Select(i => batcher.PredictAsync($"state {i}", Q(), ct: Ct)));

        Assert.Equal([8], inner.Chunks);
        Assert.Equal(Enumerable.Range(0, 8).Select(i => $"state {i}"), results.Select(r => r.Model));
    }

    [Fact]
    public async Task Different_questions_and_options_get_separate_batches()
    {
        var inner = new ChunkRecordingPredictor();
        await using var batcher = new MicroBatchingPredictor(inner, new MicroBatchOptions { MaxDelay = TimeSpan.FromSeconds(1), MaxBatchSize = 4 });

        await Task.WhenAll(
            batcher.PredictAsync("a", Q(), ct: Ct),
            batcher.PredictAsync("b", Q("Is this spam?"), ct: Ct),
            batcher.PredictAsync("c", Q(), new PredictOptions { MaxLen = 64 }, Ct),
            batcher.PredictAsync("d", Q(), ct: Ct));

        Assert.Equal([1, 1, 2], inner.Chunks.Order());
    }

    [Fact]
    public async Task A_batch_holds_at_most_MaxBatchSize_requests()
    {
        var inner = new ChunkRecordingPredictor();
        await using var batcher = new MicroBatchingPredictor(inner, new MicroBatchOptions { MaxDelay = TimeSpan.FromSeconds(1), MaxBatchSize = 3 });

        await Task.WhenAll(Enumerable.Range(0, 7).Select(i => batcher.PredictAsync($"s{i}", Q(), ct: Ct)));

        Assert.Equal(7, inner.Chunks.Sum());
        Assert.All(inner.Chunks, c => Assert.InRange(c, 1, 3));
    }

    [Fact]
    public async Task A_request_cancelled_before_its_batch_runs_is_dropped()
    {
        var inner = new ChunkRecordingPredictor();
        await using var batcher = new MicroBatchingPredictor(inner, new MicroBatchOptions { MaxDelay = TimeSpan.FromMilliseconds(500) });
        using var cts = new CancellationTokenSource();

        var kept = batcher.PredictAsync("kept", Q(), ct: Ct);
        var dropped = batcher.PredictAsync("dropped", Q(), ct: cts.Token);
        await cts.CancelAsync();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => dropped);
        Assert.Equal("kept", (await kept).Model);
        Assert.Equal([1], inner.Chunks);
    }

    [Fact]
    public async Task A_failing_batch_fails_each_of_its_requests()
    {
        await using var batcher = new MicroBatchingPredictor(new ThrowingPredictor(), new MicroBatchOptions { MaxDelay = TimeSpan.FromMilliseconds(50) });

        var calls = Enumerable.Range(0, 3).Select(i => batcher.PredictAsync($"s{i}", Q(), ct: Ct)).ToList();

        foreach (var call in calls) await Assert.ThrowsAsync<InvalidOperationException>(() => call);
    }

    [Fact]
    public async Task Routing_options_skip_batching()
    {
        var inner = new ChunkRecordingPredictor();
        await using var batcher = new MicroBatchingPredictor(inner);

        var r = await batcher.PredictAsync("routed", Q(), new RouteOptions { Checkpoint = Checkpoint.English }, Ct);

        Assert.Equal("routed", r.Model);
        Assert.Empty(inner.Chunks);
    }

    [Fact]
    public async Task Dispose_answers_queued_requests_then_refuses_new_ones()
    {
        var inner = new ChunkRecordingPredictor();
        var batcher = new MicroBatchingPredictor(inner, new MicroBatchOptions { MaxDelay = TimeSpan.FromMilliseconds(100) });
        var queued = batcher.PredictAsync("queued", Q(), ct: Ct);

        await batcher.DisposeAsync();

        Assert.Equal("queued", (await queued).Model);
        await Assert.ThrowsAsync<ObjectDisposedException>(() => batcher.PredictAsync("late", Q(), ct: Ct));
    }

    private sealed class ThrowingPredictor : ILayaPredictor
    {
        public LayaResult Predict(LayaState state, Questions questions, PredictOptions? options = null) =>
            throw new InvalidOperationException("backend failed");

        public Task<LayaResult> PredictAsync(LayaState state, Questions questions, PredictOptions? options = null, CancellationToken ct = default) =>
            Task.FromResult(Predict(state, questions, options));
    }
}
