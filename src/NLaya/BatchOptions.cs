namespace NLaya;

/// <summary>
/// Per-call settings for <see cref="LayaAgent.PredictBatch"/>, <see cref="Routing.Router.PredictBatch(IEnumerable{Routing.RouteRequest}, BatchOptions?)"/>
/// and the <see cref="LayaPredictorExtensions.PredictStreamAsync{T}(ILayaPredictor, IAsyncEnumerable{T}, Func{T, LayaState}, Questions, BatchOptions?, CancellationToken)"/> family.
/// </summary>
public sealed class BatchOptions : PredictOptions
{
    /// <summary>States a stream reads and predicts at a time when <see cref="BatchSize"/> is not set.</summary>
    public const int DefaultStreamBatchSize = 32;

    /// <summary>
    /// States per forward pass; null sends them all in one. Streams also read their input in chunks
    /// of this size (<see cref="DefaultStreamBatchSize"/> when null), which bounds their memory.
    /// </summary>
    public int? BatchSize { get; init; }

    /// <summary>
    /// Group similar-length states (within windows of eight batches) to reduce padding. Needs a
    /// <see cref="BatchSize"/> between 1 and the number of states. Results keep input order.
    /// </summary>
    public bool SortByLength { get; init; }
}
