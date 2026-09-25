namespace NLaya.ML;

/// <summary>How <see cref="LayaTransformer"/> reads rows and names its columns.</summary>
public sealed class LayaTransformerOptions
{
    /// <summary>Rows read ahead and answered per <see cref="ILayaPredictor.PredictBatch"/> call.</summary>
    public int ChunkSize { get; init; } = 256;

    /// <summary>Forward-pass settings for each chunk. By default, 32 states per pass, grouped by length.</summary>
    public BatchOptions Batch { get; init; } = new() { BatchSize = BatchOptions.DefaultStreamBatchSize, SortByLength = true };

    /// <summary>Prepended to every output column name, to keep them apart from the input's columns.</summary>
    public string OutputColumnPrefix { get; init; } = "";
}
