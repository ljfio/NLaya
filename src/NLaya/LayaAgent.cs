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

    /// <summary>The checkpoint's <c>rl_agent_config.json</c>.</summary>
    public AgentConfig Config { get; }
    /// <summary>The checkpoint's <c>encoder/config.json</c>, when it has one.</summary>
    public ModernBertConfig? EncoderConfig { get; }
    /// <summary>The checkpoint's tokenizer.</summary>
    public LayaTokenizer Tokenizer { get; }
    /// <summary>The calibration temperatures this agent applies, after clamping and language overrides.</summary>
    public TemperatureTable Temperatures { get; }

    /// <summary>The inference backend; throws once the agent is disposed.</summary>
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

    private LayaResult[] Infer(PredictContext ctx, int? batchSize, bool sortByLength)
    {
        var results = new LayaResult[ctx.States.Count];
        var ids = ctx.Questions.Keys.ToList();
        if (ids.Count == 0)
        {
            Array.Fill(results, new LayaResult(ModelId, new OrderedDictionary<string, Answer>(), Usage.Zero));
            return results;
        }
        var questions = ids.Select(id => ctx.Questions[id]).ToList();
        RunPasses(ctx.States, ids, questions, ctx.MaxLen, ctx.HeadMaxLen, batchSize, sortByLength, (pass, batch, output) =>
        {
            var act = Decoder.Softmax(output.ActLogits, output.Rows, output.ActOutputs);
            var row = 0;
            foreach (var state in pass)
            {
                var optionCounts = state.Rows.Select(r => r.Markers.Length).ToList();
                var answers = Decoder.Decode(output, act, row, ids, questions, optionCounts, Temperatures, ctx.Lang);
                var tokens = 0;
                foreach (var n in batch.Lengths.AsSpan(row, state.Rows.Count)) tokens += n;
                results[state.Index] = new LayaResult(ModelId, answers, new Usage(tokens));
                row += state.Rows.Count;
            }
        });
        return results;
    }

    /// <summary>
    /// The raw option logits for each state and question, before temperature scaling: one
    /// <c>float[]</c> per question, one entry per option (for a noul, [false, true]). This is what
    /// temperature fitting and calibration metrics need (see <see cref="Calibration.TemperatureFitter"/>).
    /// Hooks don't run, and nothing is recorded in telemetry.
    /// </summary>
    public IReadOnlyList<OrderedDictionary<string, float[]>> PredictLogits(IEnumerable<LayaState> states, Questions questions, BatchOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(states);
        ArgumentNullException.ThrowIfNull(questions);
        options ??= new BatchOptions();
        var list = states.ToList();
        var ids = questions.Keys.ToList();
        var results = new OrderedDictionary<string, float[]>[list.Count];
        for (var i = 0; i < results.Length; i++) results[i] = [];
        if (ids.Count == 0) return results;
        RunPasses(list, ids, ids.Select(id => questions[id]).ToList(), options.MaxLen, options.HeadMaxLen, options.BatchSize, options.SortByLength,
            (pass, _, output) =>
            {
                var row = 0;
                foreach (var state in pass)
                {
                    for (var j = 0; j < ids.Count; j++, row++)
                        results[state.Index][ids[j]] = output.Logits.AsSpan(row * output.MaxMarkers, state.Rows[j].Markers.Length).ToArray();
                }
            });
        return results;
    }

    /// <summary>
    /// Tokenize every state against every question, pack the rows into forward passes of at most
    /// <paramref name="batchSize"/> states, and hand each pass's output to <paramref name="read"/>. With
    /// <paramref name="sortByLength"/>, states are grouped by length within windows of eight passes to cut
    /// padding; <see cref="EncodedState.Index"/> keeps each state's input position.
    /// </summary>
    private void RunPasses(IList<LayaState> states, List<string> ids, List<Question> questions, int? maxLenOption, int? headMaxLenOption,
        int? batchSize, bool sortByLength, Action<EncodedState[], EncodedBatch, BackendOutput> read)
    {
        if (states.Count == 0) return;
        for (var i = 0; i < ids.Count; i++)
            if (questions[i].Validate() is { } err) throw new ArgumentException($"question '{ids[i]}': {err}");

        var maxLen = maxLenOption ?? Config.MaxLen;
        var headMaxLen = headMaxLenOption ?? Config.HeadMaxLen;
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

        List<EncodedState> Encode(int start)
        {
            var encoded = Enumerable.Range(start, Math.Min(window, states.Count - start))
                .Select(i => new EncodedState(i, EncodeState(states[i], encodedQuestions, maxLen)))
                .ToList();
            return sort ? encoded.OrderBy(e => e.Rows.Max(r => r.Ids.Length)).ToList() : encoded; // stable
        }

        var current = Encode(0);
        for (var start = 0; start < states.Count; start += window)
        {
            // Tokenize the next window on the thread pool while this one runs through the model, so
            // an accelerator doesn't wait on the CPU between passes.
            var nextStart = start + window;
            var next = nextStart < states.Count ? Task.Run(() => Encode(nextStart)) : null;
            try
            {
                foreach (var pass in current.Chunk(perPass))
                {
                    var batch = EncodedBatch.Collate(pass.SelectMany(s => s.Rows).ToList(), Tokenizer.PadId);
                    read(pass, batch, Backend.Run(batch));
                }
            }
            catch
            {
                // Observe the prefetch so its own failure isn't reported as unobserved.
                next?.ContinueWith(t => _ = t.Exception, TaskScheduler.Default);
                throw;
            }
            if (next is not null) current = next.GetAwaiter().GetResult();
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

    /// <summary>
    /// Run one short forward pass straight through the backend (no hooks, no telemetry), so the first
    /// real request doesn't pay for lazy setup: the CUDA context and kernel selection, or ONNX
    /// Runtime's first-run work. The DI warm-up service calls this after loading.
    /// </summary>
    public void Warmup()
    {
        var question = SequenceBuilder.EncodeQuestion(Tokenizer, WarmupQuestion, Config.HeadMaxLen);
        var row = SequenceBuilder.Assemble(Tokenizer, question, SequenceBuilder.EncodeState(Tokenizer, "warm-up"), Config.MaxLen, truncateLeft: false);
        Backend.Run(EncodedBatch.Collate([row], Tokenizer.PadId));
    }

    private static readonly Question WarmupQuestion = Question.Noul("Is this a warm-up request?");

    /// <summary>The model id and backend, as Python's <c>repr</c>.</summary>
    public override string ToString() => $"LayaAgent(model_id='{ModelId}', backend={_backend?.Name ?? "disposed"})";

    /// <summary>Free the backend (weights and device memory).</summary>
    public void Dispose()
    {
        Interlocked.Exchange(ref _backend, null)?.Dispose();
    }

    /// <summary><see cref="Dispose"/>.</summary>
    public ValueTask DisposeAsync()
    {
        Dispose();
        return ValueTask.CompletedTask;
    }
}
