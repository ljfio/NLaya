using System.Threading.Channels;

namespace NLaya;

/// <summary>
/// Collects concurrent single-state requests for up to <see cref="MicroBatchOptions.MaxDelay"/> and
/// answers those with the same questions and options in one <see cref="ILayaPredictor.PredictBatch"/>
/// call. A GPU answers a batch of 32 in little more time than one request, so a server whose callers
/// each send one state gets most of the throughput of batching without changing the callers.
/// </summary>
/// <remarks>
/// <para>
/// Batches run one at a time, in arrival order; requests that arrive while one runs form the next.
/// Answers are the same as unbatched ones (up to floating-point noise from padding). Hooks see one
/// call per batch, as with <see cref="ILayaPredictor.PredictBatch"/>. Requests whose options carry
/// routing settings (<see cref="Routing.RouteOptions"/>) skip batching and go straight to the inner predictor.
/// </para>
/// <para>
/// Disposing stops accepting requests and waits for the queued ones. The inner predictor belongs to
/// the caller and is not disposed.
/// </para>
/// </remarks>
public sealed class MicroBatchingPredictor : ILayaPredictor, IAsyncDisposable, IDisposable
{
    private readonly ILayaPredictor _inner;
    private readonly MicroBatchOptions _options;
    private readonly Channel<Pending> _queue = Channel.CreateUnbounded<Pending>(new UnboundedChannelOptions { SingleReader = true });
    private readonly Task _loop;

    /// <summary>Batch requests to <paramref name="inner"/>, usually a <see cref="LayaAgent"/> or <see cref="Routing.Router"/>.</summary>
    public MicroBatchingPredictor(ILayaPredictor inner, MicroBatchOptions? options = null)
    {
        _inner = inner ?? throw new ArgumentNullException(nameof(inner));
        _options = options ?? new MicroBatchOptions();
        ArgumentOutOfRangeException.ThrowIfLessThan(_options.MaxBatchSize, 1, nameof(options));
        ArgumentOutOfRangeException.ThrowIfLessThan(_options.MaxDelay, TimeSpan.Zero, nameof(options));
        _loop = Task.Run(RunAsync);
    }

    /// <summary>The predictor that answers the batches.</summary>
    public ILayaPredictor Inner => _inner;

    /// <summary>Queue the request and wait for its batch; <paramref name="ct"/> drops it if its batch hasn't started.</summary>
    public Task<LayaResult> PredictAsync(LayaState state, Questions questions, PredictOptions? options = null, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(state);
        ArgumentNullException.ThrowIfNull(questions);
        if (ct.IsCancellationRequested) return Task.FromCanceled<LayaResult>(ct);
        if (options is not null && options.GetType() != typeof(PredictOptions) && options is not BatchOptions)
            return _inner.PredictAsync(state, questions, options, ct);

        var pending = new Pending(state, questions, options, ct);
        ObjectDisposedException.ThrowIf(!_queue.Writer.TryWrite(pending), this);
        return pending.Result.Task;
    }

    /// <summary>Synchronous <see cref="PredictAsync"/>: blocks the calling thread until the request's batch is answered.</summary>
    public LayaResult Predict(LayaState state, Questions questions, PredictOptions? options = null) =>
        PredictAsync(state, questions, options).GetAwaiter().GetResult();

    /// <summary>Already a batch: passed straight to the inner predictor.</summary>
    public IReadOnlyList<LayaResult> PredictBatch(IEnumerable<LayaState> states, Questions questions, BatchOptions? options = null) =>
        _inner.PredictBatch(states, questions, options);

    /// <summary>Already a batch: passed straight to the inner predictor.</summary>
    public Task<IReadOnlyList<LayaResult>> PredictBatchAsync(IEnumerable<LayaState> states, Questions questions, BatchOptions? options = null,
        CancellationToken ct = default) =>
        _inner.PredictBatchAsync(states, questions, options, ct);

    private async Task RunAsync()
    {
        var reader = _queue.Reader;
        var batch = new List<Pending>(_options.MaxBatchSize);
        while (await reader.WaitToReadAsync().ConfigureAwait(false))
        {
            if (!reader.TryRead(out var first)) continue;
            batch.Add(first);
            await FillAsync(reader, batch).ConfigureAwait(false);
            Answer(batch);
            batch.Clear();
        }
    }

    /// <summary>Add requests until the batch is full or <see cref="MicroBatchOptions.MaxDelay"/> has passed since the first.</summary>
    private async Task FillAsync(ChannelReader<Pending> reader, List<Pending> batch)
    {
        using var window = new CancellationTokenSource(_options.MaxDelay);
        try
        {
            while (true)
            {
                while (batch.Count < _options.MaxBatchSize && reader.TryRead(out var p)) batch.Add(p);
                if (batch.Count >= _options.MaxBatchSize || !await reader.WaitToReadAsync(window.Token).ConfigureAwait(false)) return;
            }
        }
        catch (OperationCanceledException) when (window.IsCancellationRequested)
        {
            // The window closed: answer what has arrived.
        }
    }

    private void Answer(List<Pending> batch)
    {
        LayaTelemetry.MicroBatch(batch.Count);
        foreach (var group in batch.GroupBy(p => p.Key))
        {
            var live = new List<Pending>();
            foreach (var p in group)
            {
                if (p.Ct.IsCancellationRequested) p.Result.TrySetCanceled(p.Ct);
                else live.Add(p);
            }
            if (live.Count == 0) continue;

            var o = live[0].Options;
            try
            {
                var results = _inner.PredictBatch(live.Select(p => p.State).ToList(), live[0].Questions, new BatchOptions
                {
                    Lang = o?.Lang,
                    MaxLen = o?.MaxLen,
                    HeadMaxLen = o?.HeadMaxLen,
                    Hooks = o?.Hooks,
                    ThrowOnHookError = o?.ThrowOnHookError,
                    BatchSize = _options.PassSize,
                    SortByLength = _options.SortByLength,
                });
                if (results.Count != live.Count)
                    throw new InvalidOperationException($"PredictBatch returned {results.Count} results for {live.Count} states");
                for (var i = 0; i < live.Count; i++) live[i].Result.TrySetResult(results[i]);
            }
            catch (Exception ex)
            {
                foreach (var p in live) p.Result.TrySetException(ex);
            }
        }
    }

    /// <summary>Stop accepting requests and wait until the queued ones are answered.</summary>
    public async ValueTask DisposeAsync()
    {
        _queue.Writer.TryComplete();
        await _loop.ConfigureAwait(false);
    }

    /// <inheritdoc cref="DisposeAsync"/>
    public void Dispose() => DisposeAsync().AsTask().GetAwaiter().GetResult();

    /// <summary>A queued request. Requests with equal keys share a batch.</summary>
    private sealed class Pending(LayaState state, Questions questions, PredictOptions? options, CancellationToken ct)
    {
        public LayaState State => state;
        public Questions Questions => questions;
        public PredictOptions? Options => options;
        public CancellationToken Ct => ct;

        // Continuations run off the batching loop, so a caller can't stall the next batch.
        public TaskCompletionSource<LayaResult> Result { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);

        // Questions compare by content (callers often build the same set per request); hooks by reference.
        public BatchKey Key { get; } = new(questions.ToJson().ToJsonString(), options?.Lang, options?.MaxLen, options?.HeadMaxLen,
            options?.ThrowOnHookError, options?.Hooks);
    }

    private readonly record struct BatchKey(string Questions, string? Lang, int? MaxLen, int? HeadMaxLen, bool? ThrowOnHookError,
        IEnumerable<ILayaHook>? Hooks);
}
