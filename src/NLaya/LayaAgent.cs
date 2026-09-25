using System.Diagnostics.CodeAnalysis;

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
public sealed class LayaAgent : HookRegistry, ILayaPredictor, IDisposable, IAsyncDisposable
{
    private ILayaBackend? _backend;

    internal LayaAgent(LayaCheckpoint checkpoint, ILayaBackend backend, LayaOptions options)
        : base(options.Hooks, options.ThrowOnHookError, options.Logger)
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
            Logger.InvalidTemperatures(TemperatureTable.Min, TemperatureTable.Max, string.Join(", ", Temperatures.Rejected));
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
    [RequiresUnreferencedCode(LayaState.ReflectionMessage)]
    [RequiresDynamicCode(LayaState.ReflectionMessage)]
    public LayaResult Predict(object state, Questions questions, PredictOptions? options = null) =>
        Predict(LayaState.From(state), questions, options);

    /// <summary>
    /// The same questions over many states, packed into shared forward passes. Results are
    /// aligned with <paramref name="states"/>.
    /// </summary>
    public IReadOnlyList<LayaResult> PredictBatch(IEnumerable<LayaState> states, Questions questions, BatchOptions? options = null)
    {
        options ??= new BatchOptions();
        return PredictBatchCore(states.ToList(), questions, options, options.BatchSize, options.SortByLength).ToList();
    }

    /// <summary>
    /// <see cref="Predict(LayaState, Questions, PredictOptions?)"/> on the thread pool. Inference is
    /// CPU- or GPU-bound, so <paramref name="ct"/> only cancels a call that hasn't started yet.
    /// </summary>
    public Task<LayaResult> PredictAsync(LayaState state, Questions questions, PredictOptions? options = null, CancellationToken ct = default) =>
        Task.Run(() => Predict(state, questions, options), ct);

    /// <summary><see cref="PredictBatch"/> on the thread pool; <paramref name="ct"/> only cancels a call that hasn't started yet.</summary>
    public Task<IReadOnlyList<LayaResult>> PredictBatchAsync(IEnumerable<LayaState> states, Questions questions, BatchOptions? options = null, CancellationToken ct = default) =>
        Task.Run(() => PredictBatch(states, questions, options), ct);

    private IList<LayaResult> PredictBatchCore(List<LayaState> states, Questions questions, PredictOptions options, int? batchSize, bool sortByLength)
    {
        var active = Compose(options.Hooks);
        var raise = options.ThrowOnHookError ?? ThrowOnHookError;
        var ctx = new PredictContext(states, questions)
        {
            Model = ModelId,
            Agent = this,
            MaxLen = options.MaxLen,
            HeadMaxLen = options.HeadMaxLen,
            Lang = options.Lang,
        };
        return LayaTelemetry.Predict(ModelId, states.Count, () => RunWithHooks(active, ctx, raise, c => Infer(c, batchSize, sortByLength)));
    }

    /// <summary>
    /// Tokenize every state against every question, pack the rows into forward passes of at most
    /// <paramref name="batchSize"/> states, and decode each pass. With <paramref name="sortByLength"/>,
    /// states are grouped by length within windows of eight passes to cut padding; results keep input order.
    /// </summary>
    private IList<LayaResult> Infer(PredictContext ctx, int? batchSize, bool sortByLength)
    {
        var states = ctx.States;
        if (states.Count == 0) return [];
        var ids = ctx.Questions.Keys.ToList();
        if (ids.Count == 0)
            return states.Select(_ => new LayaResult(ModelId, new OrderedDictionary<string, Answer>(), Usage.Zero)).ToList();

        var questions = ids.Select(id => ctx.Questions[id]).ToList();
        for (var i = 0; i < ids.Count; i++)
            if (questions[i].Validate() is { } err) throw new ArgumentException($"question '{ids[i]}': {err}");

        var maxLen = ctx.MaxLen ?? Config.MaxLen;
        var headMaxLen = ctx.HeadMaxLen ?? Config.HeadMaxLen;
        var perPass = batchSize is > 0 ? batchSize.Value : states.Count;
        var sort = sortByLength && perPass > 1 && perPass < states.Count;
        var window = sort ? perPass * 8 : perPass; // bounds how many tokenized states are held at once

        var encodedQuestions = new EncodedQuestion[questions.Count];
        for (var i = 0; i < questions.Count; i++)
        {
            encodedQuestions[i] = SequenceBuilder.EncodeQuestion(Tokenizer, questions[i], headMaxLen);
            if (encodedQuestions[i].Markers.Count(m => m < maxLen) != questions[i].OptionCount)
                throw new ArgumentException($"question '{ids[i]}' options exceed head_max_len={headMaxLen}");
        }

        var results = new LayaResult[states.Count];
        for (var start = 0; start < states.Count; start += window)
        {
            var encoded = Enumerable.Range(start, Math.Min(window, states.Count - start))
                .Select(i => new EncodedState(i, EncodeState(states[i], encodedQuestions, maxLen)))
                .ToList();
            if (sort) encoded = encoded.OrderBy(e => e.Rows.Max(r => r.Ids.Length)).ToList(); // stable
            foreach (var pass in encoded.Chunk(perPass))
                RunPass(pass, ids, questions, ctx.Lang, results);
        }
        return results;
    }

    /// <summary>One forward pass over several states' question rows; writes each state's result by its input index.</summary>
    private void RunPass(EncodedState[] states, List<string> ids, List<Question> questions, string? lang, LayaResult[] results)
    {
        var batch = EncodedBatch.Collate(states.SelectMany(s => s.Rows).ToList(), Tokenizer.PadId);
        var output = Backend.Run(batch);
        var act = Decoder.Softmax(output.ActLogits, output.Rows, output.ActOutputs);
        var row = 0;
        foreach (var state in states)
        {
            var optionCounts = state.Rows.Select(r => r.Markers.Length).ToList();
            var answers = Decoder.Decode(output, act, row, ids, questions, optionCounts, Temperatures, lang);
            var tokens = 0;
            foreach (var n in batch.Lengths.AsSpan(row, state.Rows.Count)) tokens += n;
            results[state.Index] = new LayaResult(ModelId, answers, new Usage(tokens));
            row += state.Rows.Count;
        }
    }

    /// <summary>One sequence per question for <paramref name="state"/>, sharing a single tokenization of it.</summary>
    private List<EncodedItem> EncodeState(LayaState state, EncodedQuestion[] questions, int maxLen)
    {
        var stateIds = SequenceBuilder.EncodeState(Tokenizer, state);
        var rows = new List<EncodedItem>(questions.Length);
        foreach (var q in questions)
            rows.Add(SequenceBuilder.Assemble(Tokenizer, q, stateIds, maxLen, state.IsConversation));
        return rows;
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
