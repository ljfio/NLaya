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
        var ct = TestContext.Current.CancellationToken;
        var concurrent = await Task.WhenAll(states.Select(s => Task.Run(() => agent.Predict(s, questions).ToJson(), ct)));

        for (var i = 0; i < states.Count; i++) ModelParityTests.AssertResult(sequential[i], concurrent[i]);
    }
}
