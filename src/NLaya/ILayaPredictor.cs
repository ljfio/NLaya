namespace NLaya;

/// <summary>
/// Answers typed questions about a state: implemented by <see cref="LayaAgent"/> and
/// <see cref="Routing.Router"/>, so integrations (and test fakes) can take either.
/// </summary>
public interface ILayaPredictor
{
    /// <summary>Answer <paramref name="questions"/> about one <paramref name="state"/>.</summary>
    LayaResult Predict(LayaState state, Questions questions, PredictOptions? options = null);

    /// <summary>Asynchronous <see cref="Predict"/>.</summary>
    Task<LayaResult> PredictAsync(LayaState state, Questions questions, PredictOptions? options = null, CancellationToken ct = default);
}
