using System.Runtime.CompilerServices;

namespace NLaya;

/// <summary>
/// Streaming prediction for any <see cref="ILayaPredictor"/>: read the input in chunks of
/// <see cref="BatchOptions.BatchSize"/> (default <see cref="BatchOptions.DefaultStreamBatchSize"/>),
/// answer each chunk with one <see cref="ILayaPredictor.PredictBatchAsync"/> call, and yield the
/// results in input order. Memory stays bounded by the chunk, so a database cursor, a file or a queue
/// of any length can be scored. Hooks see one call per chunk, as with <c>PredictBatch</c>.
/// Cancellation is checked between chunks; a chunk already running finishes first.
/// </summary>
public static class LayaPredictorExtensions
{
    /// <summary>Answer <paramref name="questions"/> about each state, a chunk at a time.</summary>
    public static IAsyncEnumerable<LayaResult> PredictStreamAsync(this ILayaPredictor predictor, IEnumerable<LayaState> states,
        Questions questions, BatchOptions? options = null, CancellationToken ct = default) =>
        predictor.PredictStreamAsync(states.ToAsyncEnumerable(), questions, options, ct);

    /// <summary>Answer <paramref name="questions"/> about each state, a chunk at a time.</summary>
    public static async IAsyncEnumerable<LayaResult> PredictStreamAsync(this ILayaPredictor predictor, IAsyncEnumerable<LayaState> states,
        Questions questions, BatchOptions? options = null, [EnumeratorCancellation] CancellationToken ct = default)
    {
        await foreach (var p in predictor.PredictStreamAsync(states, s => s, questions, options, ct).ConfigureAwait(false))
            yield return p.Result;
    }

    /// <summary>Answer <paramref name="questions"/> about each item, keeping the item next to its result.</summary>
    public static IAsyncEnumerable<LayaPrediction<T>> PredictStreamAsync<T>(this ILayaPredictor predictor, IEnumerable<T> items,
        Func<T, LayaState> state, Questions questions, BatchOptions? options = null, CancellationToken ct = default) =>
        predictor.PredictStreamAsync(items.ToAsyncEnumerable(), state, questions, options, ct);

    /// <summary>Answer <paramref name="questions"/> about each item, keeping the item next to its result.</summary>
    public static async IAsyncEnumerable<LayaPrediction<T>> PredictStreamAsync<T>(this ILayaPredictor predictor, IAsyncEnumerable<T> items,
        Func<T, LayaState> state, Questions questions, BatchOptions? options = null, [EnumeratorCancellation] CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(predictor);
        ArgumentNullException.ThrowIfNull(items);
        ArgumentNullException.ThrowIfNull(state);
        ArgumentNullException.ThrowIfNull(questions);
        var size = options?.BatchSize is > 0 ? options.BatchSize.Value : BatchOptions.DefaultStreamBatchSize;
        await foreach (var chunk in items.Chunk(size).WithCancellation(ct).ConfigureAwait(false))
        {
            ct.ThrowIfCancellationRequested();
            var results = await predictor.PredictBatchAsync(chunk.Select(state).ToList(), questions, options, ct).ConfigureAwait(false);
            if (results.Count != chunk.Length)
                throw new InvalidOperationException($"PredictBatchAsync returned {results.Count} results for {chunk.Length} states");
            for (var i = 0; i < chunk.Length; i++)
                yield return new LayaPrediction<T>(chunk[i], results[i]);
        }
    }
}
