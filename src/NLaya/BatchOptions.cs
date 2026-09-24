namespace NLaya;

/// <summary>Per-call settings for <see cref="LayaAgent.PredictBatch"/>.</summary>
public sealed class BatchOptions : PredictOptions
{
    /// <summary>States per forward pass; null sends them all in one.</summary>
    public int? BatchSize { get; init; }

    /// <summary>
    /// Group similar-length states (within windows of eight batches) to reduce padding. Needs a
    /// <see cref="BatchSize"/> between 1 and the number of states. Results keep input order.
    /// </summary>
    public bool SortByLength { get; init; }
}
