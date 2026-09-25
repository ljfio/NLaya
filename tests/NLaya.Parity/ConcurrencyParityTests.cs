using System.Text.Json.Nodes;

namespace NLaya.Parity;

/// <summary>
/// One agent answering many requests at once gives the same answers as answering them one by one.
/// The inputs span more lengths than the old per-length RoPE cache held, which used to dispose tables
/// another thread was still reading.
/// </summary>
[Collection(ParityCollection.Name)]
public class ConcurrencyParityTests(ParityFixture fx)
{
    [Theory]
    [InlineData("torchsharp", "multilingual")]
    [InlineData("onnx", "multilingual")]
    public async Task Concurrent_predictions_match_sequential_ones(string backend, string model)
    {
        var agent = fx.Agent(backend, model);
        var questions = Presets.Triage();
        var states = Enumerable.Range(1, 40)
            .Select(i => (LayaState)string.Join(' ', Enumerable.Repeat("I was charged twice for invoice 4411.", i)))
            .ToList();

        var sequential = states.Select(s => agent.Predict(s, questions).ToJson()).ToList();
        // Eight callers at a time still interleave lengths; forty at once starved a 2-core CI runner of memory.
        var concurrent = new JsonObject[states.Count];
        var options = new ParallelOptions { MaxDegreeOfParallelism = 8, CancellationToken = TestContext.Current.CancellationToken };
        await Parallel.ForEachAsync(Enumerable.Range(0, states.Count), options, (i, _) =>
        {
            concurrent[i] = agent.Predict(states[i], questions).ToJson();
            return ValueTask.CompletedTask;
        });

        for (var i = 0; i < states.Count; i++) ModelParityTests.AssertResult(sequential[i], concurrent[i]);
    }
}
