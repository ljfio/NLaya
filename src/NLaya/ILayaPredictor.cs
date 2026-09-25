namespace NLaya;

/// <summary>
/// Answers typed questions about a state: implemented by <see cref="LayaAgent"/> and
/// <see cref="Routing.Router"/>, so integrations (and test fakes) can take either.
/// Streaming lives in <see cref="LayaPredictorExtensions"/>.
/// </summary>
public interface ILayaPredictor
{
    /// <summary>Answer <paramref name="questions"/> about one <paramref name="state"/>.</summary>
    LayaResult Predict(LayaState state, Questions questions, PredictOptions? options = null);

    /// <summary>Asynchronous <see cref="Predict"/>.</summary>
    Task<LayaResult> PredictAsync(LayaState state, Questions questions, PredictOptions? options = null, CancellationToken ct = default);

    /// <summary>
    /// The same questions over many states, results aligned with <paramref name="states"/>.
    /// <see cref="LayaAgent"/> shares forward passes; <see cref="Routing.Router"/> also loads each
    /// checkpoint once. The default answers one state at a time.
    /// </summary>
    IReadOnlyList<LayaResult> PredictBatch(IEnumerable<LayaState> states, Questions questions, BatchOptions? options = null) =>
        states.Select(s => Predict(s, questions, options)).ToList();

    /// <summary>Asynchronous <see cref="PredictBatch"/>.</summary>
    Task<IReadOnlyList<LayaResult>> PredictBatchAsync(IEnumerable<LayaState> states, Questions questions, BatchOptions? options = null, CancellationToken ct = default) =>
        Task.Run(() => PredictBatch(states, questions, options), ct);
}
