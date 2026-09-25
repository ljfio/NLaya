using NLaya.Hub;
using NLaya.Tests.ExtensionsAI;

namespace NLaya.Tests.Batching;

public class StreamTests
{
    private static readonly Questions Qs = new() { ["q"] = Question.Noul("Is it?") };

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    [Fact]
    public async Task Stream_reads_default_chunks_and_keeps_items_with_their_results()
    {
        var p = new ChunkRecordingPredictor();
        var items = Enumerable.Range(0, 70).ToList();

        var got = await p.PredictStreamAsync(items, i => $"item {i}", Qs, ct: Ct).ToListAsync(Ct);

        Assert.Equal([BatchOptions.DefaultStreamBatchSize, BatchOptions.DefaultStreamBatchSize, 6], p.Chunks);
        Assert.Equal(items, got.Select(x => x.Item));
        Assert.All(got, x => Assert.Equal($"item {x.Item}", x.Result.Model));
    }

    [Fact]
    public async Task BatchSize_sets_the_chunk()
    {
        var p = new ChunkRecordingPredictor();
        var states = Enumerable.Range(0, 25).Select(i => (LayaState)$"s{i}").ToList();

        var got = await p.PredictStreamAsync(states, Qs, new BatchOptions { BatchSize = 10 }, Ct).ToListAsync(Ct);

        Assert.Equal([10, 10, 5], p.Chunks);
        Assert.Equal(states.Select(s => s.Serialize()), got.Select(r => r.Model));
    }

    [Fact]
    public async Task Stream_pulls_only_one_chunk_ahead()
    {
        var pulled = 0;
        async IAsyncEnumerable<LayaState> Source()
        {
            for (var i = 0; i < 1000; i++)
            {
                await Task.Yield();
                pulled++;
                yield return $"s{i}";
            }
        }

        await foreach (var _ in new ChunkRecordingPredictor().PredictStreamAsync(Source(), Qs, new BatchOptions { BatchSize = 5 }, Ct))
            break;

        Assert.InRange(pulled, 5, 6);
    }

    [Fact]
    public async Task Cancellation_stops_between_chunks()
    {
        var p = new ChunkRecordingPredictor();
        using var cts = CancellationTokenSource.CreateLinkedTokenSource(Ct);
        var states = Enumerable.Range(0, 20).Select(i => (LayaState)$"s{i}");

        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () =>
        {
            await foreach (var _ in p.PredictStreamAsync(states, Qs, new BatchOptions { BatchSize = 5 }, cts.Token))
                cts.Cancel();
        });
        Assert.Equal([5], p.Chunks);
    }

    [Fact]
    public async Task Empty_input_makes_no_calls()
    {
        var p = new ChunkRecordingPredictor();
        Assert.Empty(await p.PredictStreamAsync(Array.Empty<LayaState>(), Qs, ct: Ct).ToListAsync(Ct));
        Assert.Empty(p.Chunks);
    }

    [Fact]
    public async Task A_predictor_with_only_Predict_streams_through_the_default_batch()
    {
        var fake = new FakePredictor((_, _) => new() { ["q"] = FakePredictor.Noul(0.2) });
        var got = await fake.PredictStreamAsync(["a", "b", "c"], s => s, Qs, ct: Ct).ToListAsync(Ct);

        Assert.Equal(["a", "b", "c"], fake.Calls.Select(c => c.Text));
        Assert.Equal(["a", "b", "c"], got.Select(x => x.Item));
    }

    [Fact]
    public async Task Agent_stream_matches_PredictBatch_and_hooks_see_one_call_per_chunk()
    {
        if (HfCache.Snapshot(Laya.MultilingualModel) is null)
            Assert.Skip($"{Laya.MultilingualModel} is not cached; run: {HfCache.DownloadCommand(Laya.MultilingualModel)}");
        var calls = new List<int>();
        using var agent = Laya.Load(Laya.MultilingualModel, o =>
        {
            o.Backend = new LengthBackendFactory();
            o.Hooks.Add(LayaHooks.OnStart(c => calls.Add(c.States.Count)));
        });
        var states = Enumerable.Range(0, 20)
            .Select(i => (LayaState)(string.Join(' ', Enumerable.Repeat("I was charged twice for invoice 4411", 1 + i * 7 % 5)) + $" #{i}"))
            .ToList();

        var batch = agent.PredictBatch(states, Presets.Triage()).Select(r => r.ToJsonString()).ToList();
        calls.Clear();
        var streamed = await agent.PredictStreamAsync(states, Presets.Triage(), new BatchOptions { BatchSize = 6, SortByLength = true }, Ct)
            .Select(r => r.ToJsonString()).ToListAsync(Ct);

        Assert.True(batch.Distinct().Count() > 1, "the length backend should tell states apart");
        Assert.Equal(batch, streamed);
        Assert.Equal([6, 6, 6, 2], calls);
    }
}
