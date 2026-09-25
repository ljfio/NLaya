namespace NLaya;

/// <summary>Settings for <see cref="MicroBatchingPredictor"/>.</summary>
public sealed class MicroBatchOptions
{
    /// <summary>
    /// How long the first request of a batch waits for others to join it. A few milliseconds is
    /// usually enough to fill a batch under load while adding little latency when idle. Default 2 ms.
    /// </summary>
    public TimeSpan MaxDelay { get; set; } = TimeSpan.FromMilliseconds(2);

    /// <summary>Most requests answered by one <see cref="ILayaPredictor.PredictBatch"/> call. Default 32.</summary>
    public int MaxBatchSize { get; set; } = 32;

    /// <summary>
    /// States per forward pass inside a batch (<see cref="BatchOptions.BatchSize"/>); null runs the
    /// whole batch in one pass. Lower it if a full batch of long inputs doesn't fit in GPU memory.
    /// </summary>
    public int? PassSize { get; set; }

    /// <summary>Group similar-length requests within a batch (<see cref="BatchOptions.SortByLength"/>); needs <see cref="PassSize"/>.</summary>
    public bool SortByLength { get; set; }
}
