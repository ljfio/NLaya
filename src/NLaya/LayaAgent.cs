using Microsoft.Extensions.Logging;
using NLaya.Backends;
using NLaya.Calibration;
using NLaya.Config;
using NLaya.Sequences;
using NLaya.Tokenization;

namespace NLaya;

/// <summary>
/// A loaded Laya checkpoint: typed questions in, calibrated typed answers out, in one forward
/// pass. Port of Python <c>laya.Agent</c>. Thread-safe; create with <see cref="Laya.LoadAsync"/>.
/// </summary>
public sealed class LayaAgent : HookRegistry, IDisposable, IAsyncDisposable
{
    public const string ResultModelName = "laya-rl-agent";

    private ILayaBackend? _backend;

    internal LayaAgent(LayaCheckpoint checkpoint, ILayaBackend backend, LayaOptions options)
        : base(options.Hooks, options.HooksRaise, options.Logger)
    {
        ModelId = checkpoint.ModelId;
        Directory = checkpoint.Directory;
        Config = checkpoint.Config;
        EncoderConfig = checkpoint.EncoderConfig;
        Tokenizer = checkpoint.Tokenizer;
        _backend = backend;
        Temperatures = new TemperatureTable(Config.Temperature, Config.TemperatureByOptions,
            options.LangTemperatures.ToDictionary(kv => kv.Key, kv => kv.Value));
        if (Temperatures.Rejected.Count > 0)
            Logger.LogWarning(
                "laya: this checkpoint ships invalid temperatures or values outside [{Min}, {Max}]; using {Values}. " +
                "Treat confidence from the affected entries as uncalibrated.",
                TemperatureTable.Min, TemperatureTable.Max, string.Join(", ", Temperatures.Rejected));
    }

    /// <summary>The Hub id or local path this agent was loaded from.</summary>
    public string ModelId { get; }

    /// <summary>The local checkpoint directory.</summary>
    public string Directory { get; }

    public AgentConfig Config { get; }
    public ModernBertConfig? EncoderConfig { get; }
    public LayaTokenizer Tokenizer { get; }
    public TemperatureTable Temperatures { get; }

    public ILayaBackend Backend => _backend ?? throw new ObjectDisposedException(nameof(LayaAgent));

    // ---------------------------------------------------------------- predict

    /// <summary>Answer <paramref name="questions"/> about one <paramref name="state"/> in a single forward pass.</summary>
    public LayaResult Predict(LayaState state, Questions questions, PredictOptions? options = null) =>
        PredictBatchCore([state], questions, options ?? new PredictOptions(), batchSize: null, sortByLength: false)[0];

    /// <summary>Answer questions about any object (serialized to JSON with System.Text.Json).</summary>
    public LayaResult Predict(object state, Questions questions, PredictOptions? options = null) =>
        Predict(LayaState.From(state), questions, options);

    /// <summary>Alias of <see cref="Predict(LayaState, Questions, PredictOptions?)"/> (Python <c>system_one</c>).</summary>
    public LayaResult SystemOne(LayaState state, Questions questions, PredictOptions? options = null) =>
        Predict(state, questions, options);

    /// <summary>
    /// The same questions over many states, packed into shared forward passes. Results are
    /// aligned with <paramref name="states"/>.
    /// </summary>
    public IReadOnlyList<LayaResult> PredictBatch(IEnumerable<LayaState> states, Questions questions, BatchOptions? options = null)
    {
        options ??= new BatchOptions();
        return PredictBatchCore(states.ToList(), questions, options, options.BatchSize, options.SortByLength).ToList();
    }

    public Task<LayaResult> PredictAsync(LayaState state, Questions questions, PredictOptions? options = null, CancellationToken ct = default) =>
        Task.Run(() => Predict(state, questions, options), ct);

    public Task<IReadOnlyList<LayaResult>> PredictBatchAsync(IEnumerable<LayaState> states, Questions questions, BatchOptions? options = null, CancellationToken ct = default) =>
        Task.Run(() => PredictBatch(states, questions, options), ct);

    private IList<LayaResult> PredictBatchCore(IList<LayaState> states, Questions questions, PredictOptions options, int? batchSize, bool sortByLength)
    {
        var active = Compose(options.Hooks);
        var raise = options.HooksRaise ?? HooksRaise;
        var ctx = new PredictContext(states, questions)
        {
            Model = ModelId,
            Agent = this,
            MaxLen = options.MaxLen,
            HeadMaxLen = options.HeadMaxLen,
            Lang = options.Lang,
        };
        return RunWithHooks(active, ctx, raise, c => Infer(c, batchSize, sortByLength));
    }

    private IList<LayaResult> Infer(PredictContext ctx, int? batchSize, bool sortByLength)
    {
        var states = ctx.States;
        var questions = ctx.Questions;
        if (states.Count == 0) return [];
        var ids = questions.Keys.ToList();
        if (ids.Count == 0)
            return states.Select(_ => new LayaResult(ResultModelName, new OrderedMap<Answer>(), Usage.Zero)).ToList();

        var qs = ids.Select(id => questions[id]).ToList();
        for (var i = 0; i < ids.Count; i++)
            if (qs[i].Validate() is { } err) throw new ArgumentException($"question '{ids[i]}': {err}");

        var maxLen = ctx.MaxLen ?? Config.MaxLen;
        var headMaxLen = ctx.HeadMaxLen ?? Config.HeadMaxLen;
        var chunk = batchSize is > 0 ? batchSize.Value : states.Count;
        var reorder = sortByLength && chunk > 1 && chunk < states.Count;
        var window = reorder ? chunk * 8 : chunk;

        var results = new List<LayaResult>(states.Count);
        for (var start = 0; start < states.Count; start += window)
        {
            var part = states.Skip(start).Take(window).ToList();
            var encoded = part.Select(s => EncodeState(s, ids, qs, maxLen, headMaxLen)).ToList();
            var order = Enumerable.Range(0, encoded.Count).ToList();
            if (reorder) order = order.OrderBy(i => encoded[i].Max(it => it.Ids.Length)).ToList(); // stable
            var windowResults = new LayaResult[encoded.Count];
            for (var offset = 0; offset < order.Count; offset += chunk)
            {
                var indices = order.Skip(offset).Take(chunk).ToList();
                var items = indices.SelectMany(i => encoded[i]).ToList();
                var batch = EncodedBatch.Collate(items, Tokenizer.PadId);
                var output = Backend.Run(batch);
                var act = Decoder.Softmax(output.ActLogits, output.Rows, output.ActOutputs);
                var row = 0;
                foreach (var index in indices)
                {
                    var n = encoded[index].Count;
                    var tokens = 0;
                    for (var r = row; r < row + n; r++) tokens += batch.Lengths[r];
                    var answers = Decoder.Decode(output, act, row, ids, qs, encoded[index].Select(e => e.Markers.Length).ToList(),
                        Temperatures, ctx.Lang);
                    windowResults[index] = new LayaResult(ResultModelName, answers, new Usage(tokens));
                    row += n;
                }
            }
            results.AddRange(windowResults);
        }
        return results;
    }

    private List<EncodedItem> EncodeState(LayaState state, List<string> ids, List<Question> qs, int maxLen, int headMaxLen)
    {
        var stateIds = SequenceBuilder.EncodeState(Tokenizer, state);
        var items = new List<EncodedItem>(qs.Count);
        for (var i = 0; i < qs.Count; i++)
        {
            var item = SequenceBuilder.Build(Tokenizer, stateIds, qs[i], maxLen, headMaxLen, state.IsConversation);
            if (item.Markers.Length != qs[i].OptionCount)
                throw new ArgumentException($"question '{ids[i]}' options exceed head_max_len={headMaxLen}");
            items.Add(item);
        }
        return items;
    }

    public override string ToString() => $"LayaAgent(model_id='{ModelId}', backend={_backend?.Name ?? "disposed"})";

    public void Dispose()
    {
        Interlocked.Exchange(ref _backend, null)?.Dispose();
    }

    public ValueTask DisposeAsync()
    {
        Dispose();
        return ValueTask.CompletedTask;
    }
}
