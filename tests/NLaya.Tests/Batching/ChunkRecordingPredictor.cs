namespace NLaya.Tests.Batching;

/// <summary>A predictor that records each batch it is given and names every result after its state.</summary>
internal sealed class ChunkRecordingPredictor : ILayaPredictor
{
    public List<int> Chunks { get; } = [];

    public LayaResult Predict(LayaState state, Questions questions, PredictOptions? options = null) =>
        new(state.Serialize(), new OrderedDictionary<string, Answer>(), new Usage(1));

    public Task<LayaResult> PredictAsync(LayaState state, Questions questions, PredictOptions? options = null, CancellationToken ct = default) =>
        Task.FromResult(Predict(state, questions, options));

    public IReadOnlyList<LayaResult> PredictBatch(IEnumerable<LayaState> states, Questions questions, BatchOptions? options = null)
    {
        var list = states.ToList();
        lock (Chunks) Chunks.Add(list.Count);
        return list.Select(s => Predict(s, questions, options)).ToList();
    }
}
