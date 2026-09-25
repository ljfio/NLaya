using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Runtime.ExceptionServices;

using NLaya.Lang;

namespace NLaya.Routing;

/// <summary>
/// Sends each request to the Laya checkpoint that can read it, loading checkpoints lazily (port of
/// <c>laya.Router</c>): non-Latin scripts and non-English Latin text go to <c>multilingual</c>, English
/// to <c>english</c>, and <c>typed-decisions</c> only when asked for.
/// </summary>
public sealed class Router : HookRegistry, ILayaPredictor, IDisposable
{
    public const string BundleRepo = "convaiinnovations/laya";

    public static readonly IReadOnlyDictionary<string, CheckpointSpec> DefaultModels = new Dictionary<string, CheckpointSpec>
    {
        ["english"] = new(BundleRepo),
        ["multilingual"] = new(BundleRepo, "multilingual"),
        ["typed-decisions"] = new(BundleRepo, "typed-decisions"),
    };

    public static readonly IReadOnlyDictionary<string, CheckpointSpec> StandaloneModels = new Dictionary<string, CheckpointSpec>
    {
        ["english"] = new("convaiinnovations/laya"),
        ["multilingual"] = new("convaiinnovations/laya-multilingual"),
        ["typed-decisions"] = new("convaiinnovations/laya-typed-decisions"),
    };

    private static readonly Dictionary<string, string> Aliases = new()
    {
        ["en"] = "english",
        ["laya"] = "english",
        ["default"] = "english",
        ["multi"] = "multilingual",
        ["ml"] = "multilingual",
        ["laya-multilingual"] = "multilingual",
        ["typed"] = "typed-decisions",
        ["typed_decisions"] = "typed-decisions",
        ["laya-typed-decisions"] = "typed-decisions",
        ["decisions"] = "typed-decisions",
    };

    private static readonly (string Name, string[] Ids)[] Workflows =
    [
        ("agent_trace_observability", ["action", "needs_review", "outcome", "risk", "urgency"]),
        ("customer_service", ["action", "category", "churn_risk", "needs_human", "urgency"]),
        ("invoice_processing", ["discrepancy_severity", "disposition", "duplicate", "matches_order", "urgency"]),
        ("security_incidents", ["credential_compromise", "disposition", "severity", "true_positive", "urgency"]),
    ];

    private readonly RouterOptions _o;
    private readonly Dictionary<string, CheckpointSpec> _models;
    private readonly Dictionary<string, LayaAgent> _agents = new();
    private readonly HashSet<string> _attached = new();
    private readonly List<string> _order = new(); // least recently used first
    private readonly Lock _lock = new();
    private int _maxLoaded;

    public Router(RouterOptions? options = null) : base((options ??= new()).Hooks, options.HooksRaise, options.Logger)
    {
        _o = options;
        _models = new Dictionary<string, CheckpointSpec>(options.StandaloneRepos ? StandaloneModels : DefaultModels);
        foreach (var (k, v) in options.Models) _models[NormaliseName(k)] = v;
        _maxLoaded = Math.Max(1, options.MaxLoaded);
        Default = NormaliseName(options.Default);
    }

    /// <summary>A router whose checkpoints all load with <paramref name="configure"/>, e.g. <c>o => o.UseTorchSharp()</c>.</summary>
    public Router(Action<LayaOptions> configure) : this(new RouterOptions { ConfigureAgent = (_, o) => configure(o) }) { }

    public string Default { get; }
    public IReadOnlyDictionary<string, CheckpointSpec> Models => _models;

    public IReadOnlyList<string> Loaded
    {
        get { lock (_lock) return _order.ToList(); }
    }

    /// <summary>"english", "multilingual" or "typed-decisions" for a name or alias.</summary>
    public static string NormaliseName(string name)
    {
        var key = name.Trim().ToLowerInvariant();
        key = Aliases.GetValueOrDefault(key, key);
        if (!DefaultModels.ContainsKey(key))
            throw new ArgumentException($"unknown model {PyStr.Repr(name)}; choose one of ['english', 'multilingual', 'typed-decisions'] " +
                $"(or an alias: [{string.Join(", ", Aliases.Keys.Order(StringComparer.Ordinal).Select(PyStr.Repr))}])");
        return key;
    }

    /// <summary>The typed-decisions workflow whose exact question-id set this is, else null.</summary>
    public static string? MatchTypedDecisionsWorkflow(IEnumerable<string>? questionIds)
    {
        var ids = (questionIds ?? []).ToHashSet();
        return Workflows.FirstOrDefault(w => ids.SetEquals(w.Ids)).Name;
    }

    // ---------------------------------------------------------------- loading

    /// <summary>The agent for a checkpoint, loading it on first use.</summary>
    public LayaAgent Load(string name)
    {
        var key = NormaliseName(name);
        LayaAgent agent;
        List<string> evicted;
        lock (_lock)
        {
            if (_agents.TryGetValue(key, out var existing))
            {
                Touch(key);
                return existing;
            }
            var spec = _models[key];
            agent = Laya.Load(spec.Repo, o =>
            {
                o.Subfolder = spec.Subfolder;
                o.Logger ??= _o.Logger;
                _o.ConfigureAgent?.Invoke(key, o);
            });
            _agents[key] = agent;
            _order.Add(key);
            evicted = EvictLocked();
        }
        // Lifecycle hooks run outside the lock, so a hook may call back into the router.
        Lifecycle(evicted, evict: true);
        Lifecycle([key], evict: false, agent);
        return agent;
    }

    /// <summary>Register an agent you already built instead of loading a second copy.</summary>
    public LayaAgent Attach(string name, LayaAgent agent)
    {
        var key = NormaliseName(name);
        lock (_lock)
        {
            _agents[key] = agent;
            _attached.Add(key);
            Touch(key);
            _maxLoaded = Math.Max(_maxLoaded, _agents.Count);
        }
        return agent;
    }

    /// <summary>Load checkpoints up front (all of them by default) so no request pays for a load.</summary>
    public Router Preload(IEnumerable<string>? names = null)
    {
        var keys = (names ?? _models.Keys).Select(NormaliseName).Distinct().ToList();
        lock (_lock) _maxLoaded = Math.Max(_maxLoaded, keys.Union(_agents.Keys).Count());
        foreach (var k in keys) Load(k);
        return this;
    }

    public Task<Router> PreloadAsync(IEnumerable<string>? names = null, CancellationToken ct = default) =>
        System.Threading.Tasks.Task.Run(() => Preload(names), ct);

    /// <summary>Drop one checkpoint, or all. Memory is reclaimed once in-flight calls finish.</summary>
    public void Unload(string? name = null)
    {
        List<string> freed;
        lock (_lock)
        {
            if (name is null)
            {
                freed = _order.ToList();
                _agents.Clear();
                _order.Clear();
            }
            else
            {
                var key = NormaliseName(name);
                freed = _agents.Remove(key) ? [key] : [];
                _order.Remove(key);
            }
        }
        GC.Collect();
        Lifecycle(freed, evict: true);
    }

    private void Touch(string key)
    {
        _order.Remove(key);
        _order.Add(key);
    }

    private List<string> EvictLocked()
    {
        var evicted = new List<string>();
        while (_order.Count > _maxLoaded)
        {
            var victim = _order[0];
            _order.RemoveAt(0);
            if (_agents.Remove(victim)) evicted.Add(victim);
        }
        // Evicted agents are not disposed: a concurrent call may still hold one. The GC frees them.
        if (evicted.Count > 0) GC.Collect();
        return evicted;
    }

    private void Lifecycle(IEnumerable<string> names, bool evict, LayaAgent? agent = null)
    {
        foreach (var n in names)
        {
            var ctx = new PredictContext([], new Questions()) { Model = n, Agent = agent, Router = this };
            Dispatch(Compose(null), evict ? h => h.OnEvict(ctx) : h => h.OnLoad(ctx), HooksRaise, evict ? "OnEvict" : "OnLoad");
        }
    }

    // ---------------------------------------------------------------- routing

    /// <summary>Choose a checkpoint without loading or running anything; <c>OnRoute</c> hooks may replace the decision.</summary>
    public RouteDecision Route(LayaState state, Questions? questions = null, RouteOptions? options = null)
    {
        options ??= new RouteOptions();
        var ctx = new PredictContext([state], questions ?? new Questions())
        {
            Decision = Decide(state, questions, options),
            Router = this,
        };
        Dispatch(Compose(options.Hooks), h => h.OnRoute(ctx), options.HooksRaise ?? HooksRaise, nameof(ILayaHook.OnRoute));
        return ctx.Decision!;
    }

    private RouteDecision Decide(LayaState state, Questions? questions, RouteOptions o)
    {
        RouteDecision To(string key, string reason, LanguageDetection? det = null, string? wf = null) =>
            new(key, _models[key].ToString(), reason, det, wf);

        if (o.Model is not null) return To(NormaliseName(o.Model), $"explicit model={PyStr.Repr(o.Model)}");
        if (o.Task is not null)
        {
            var key = NormaliseName(o.Task.ToLowerInvariant().Replace('-', '_') == "typed_decisions" ? "typed-decisions" : o.Task);
            return To(key, $"explicit task={PyStr.Repr(o.Task)}");
        }

        var workflow = MatchTypedDecisionsWorkflow(questions?.Keys);
        if (workflow is not null && _o.AutoTaskDetection)
            return To("typed-decisions", $"question ids match the {PyStr.Repr(workflow)} typed-decisions workflow", null, workflow);

        if (o.Lang is not null && EnglishFromCode(o.Lang) is { } byLang)
            return To(byLang ? "english" : "multilingual", $"explicit lang={PyStr.Repr(o.Lang)}", null, workflow);

        foreach (var (source, hint) in new[] { ("lang_guess", o.LangGuess), ("Router(lang_guess=...)", _o.LangGuess) })
        {
            if (hint is not null && EnglishFromCode(hint(state)) is { } en)
                return To(en ? "english" : "multilingual",
                    $"{source}: the caller identified this as {(en ? "English" : "non-English")} text", null, workflow);
        }

        var det = LanguageDetector.Analyse(state);
        string model, reason;
        if (det.Script == "unknown")
            (model, reason) = (Default, $"no letters detected in state; using default ({Default})");
        else if (det.Script != "latin")
            (model, reason) = ("multilingual",
                $"non-Latin script ({det.Script}, {PyStr.F0(100 * det.NonLatinFraction)}% of letters); the English checkpoint cannot read it");
        else if (!det.IsEnglish)
            (model, reason) = ("multilingual",
                det.MixedSegment is not null
                    ? $"Latin script, mostly English, but a line or field reads as {PyStr.Repr(det.Language!)} ({PyStr.Repr(PyStr.Take(det.MixedSegment, 60))}); the English checkpoint cannot read it"
                    : det.Language is not null
                        ? $"Latin script but language looks like {PyStr.Repr(det.Language)}, not English"
                        : $"Latin script, language not identified but {PyStr.F0(100 * det.DiacriticRate)}% non-English letters; not safe for the English checkpoint");
        else if (det.LanguageUndecided)
            (model, reason) = (Default, $"Latin script, language not identified and no non-English letters; using default ({Default})");
        else
            (model, reason) = ("english", "English Latin text");
        return To(model, reason, det, workflow);
    }

    /// <summary>True/false for a language code ("en", "en_US.UTF-8", "pt-BR"); null when it names no language.</summary>
    internal static bool? EnglishFromCode(string? code)
    {
        if (code is null) return null;
        code = code.Trim().ToLowerInvariant();
        if (code.Length == 0) return null;
        var primary = code.Split('.', 2)[0].Replace('_', '-').Split('-', 2)[0];
        if (primary.Length == 0 || primary is "c" or "posix" or "und" or "zxx" or "mul") return null;
        return primary is "en" or "eng" or "english";
    }

    // ---------------------------------------------------------------- predict

    /// <summary>Route, then answer on the chosen checkpoint. The result's <see cref="LayaResult.Routing"/> records the decision.</summary>
    public LayaResult Predict(LayaState state, Questions questions, RouteOptions? options = null)
    {
        options ??= new RouteOptions();
        var decision = Route(state, questions, options);
        var agent = Load(decision.Model);
        var lang = options.Lang ?? decision.Detection?.Language;
        var ctx = new PredictContext([state], questions)
        {
            Decision = decision,
            Model = decision.Model,
            Agent = agent,
            Router = this,
            MaxLen = options.MaxLen,
            HeadMaxLen = options.HeadMaxLen,
            Lang = lang,
        };
        var results = RunWithHooks(Compose(options.Hooks), ctx, options.HooksRaise ?? HooksRaise, c =>
            [agent.Predict(c.States[0], c.Questions, new PredictOptions { Lang = c.Lang, MaxLen = c.MaxLen, HeadMaxLen = c.HeadMaxLen })]);
        return results[0].WithRouting(decision);
    }

    /// <summary>Route and answer questions about any object (serialized to JSON with System.Text.Json).</summary>
    [RequiresUnreferencedCode(LayaState.ReflectionMessage)]
    [RequiresDynamicCode(LayaState.ReflectionMessage)]
    public LayaResult Predict(object state, Questions questions, RouteOptions? options = null) =>
        Predict(LayaState.From(state), questions, options);

    public Task<LayaResult> PredictAsync(LayaState state, Questions questions, RouteOptions? options = null, CancellationToken ct = default) =>
        System.Threading.Tasks.Task.Run(() => Predict(state, questions, options), ct);

    // ---------------------------------------------------------------- batch

    /// <summary>
    /// Route a heterogeneous batch without loading anything (Python <c>route_batch</c>), so a caller
    /// can inspect or aggregate the decisions first. Decisions keep input order.
    /// </summary>
    public IReadOnlyList<RouteDecision> RouteBatch(IEnumerable<RouteRequest> requests) =>
        RouteBatchCore(requests.ToList(), null, null);

    private List<RouteDecision> RouteBatchCore(List<RouteRequest> requests, IEnumerable<ILayaHook>? hooks, bool? hooksRaise)
    {
        var decisions = new List<RouteDecision>(requests.Count);
        for (var i = 0; i < requests.Count; i++)
        {
            var r = requests[i] ?? throw new ArgumentException($"request {i} is null", nameof(requests));
            if (r.State is null) throw new ArgumentException($"request {i} is missing required key 'state'", nameof(requests));
            if (r.Questions is null) throw new ArgumentException($"request {i} is missing required key 'questions'", nameof(requests));
            decisions.Add(Route(r.State, r.Questions, r.ToRouteOptions(hooks, hooksRaise)));
        }
        return decisions;
    }

    /// <summary>
    /// Route and answer a heterogeneous batch with as few loads as possible (Python <c>predict_batch</c>).
    /// Requests are grouped by checkpoint (in first-appearance order, so each loads at most once),
    /// then by question set, token budget and (for checkpoints with per-language temperatures)
    /// language, and each group goes through <see cref="LayaAgent.PredictBatch"/> to share forward
    /// passes. Results come back in input order with <see cref="LayaResult.Routing"/> set.
    /// </summary>
    /// <remarks>
    /// Router hooks run per request, as <see cref="Predict(LayaState, Questions, RouteOptions?)"/> runs
    /// them: a start hook can rewrite or skip its request before the grouping, and an end hook sees
    /// its result. If a group fails, every started request without a result gets <c>OnError</c>, then
    /// every started request gets <c>OnPredictEnd</c>, before the exception propagates.
    /// <paramref name="options"/> gives the batch size, length sorting, hooks and token budget for
    /// every request; the language comes from each <see cref="RouteRequest.Lang"/>, not
    /// <see cref="PredictOptions.Lang"/>.
    /// </remarks>
    public IReadOnlyList<LayaResult> PredictBatch(IEnumerable<RouteRequest> requests, BatchOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(requests);
        options ??= new BatchOptions();
        var reqs = requests.ToList();
        var raise = options.HooksRaise ?? HooksRaise;
        var decisions = RouteBatchCore(reqs, options.Hooks, options.HooksRaise);
        if (decisions.Count == 0) return [];

        // First-appearance order keeps loads deterministic and collapses an interleaved workload to
        // one load per checkpoint for this call.
        var groups = new OrderedDictionary<string, List<int>>(StringComparer.Ordinal);
        for (var i = 0; i < decisions.Count; i++)
        {
            if (!groups.TryGetValue(decisions[i].Model, out var indices)) groups[decisions[i].Model] = indices = [];
            indices.Add(i);
        }

        var active = Compose(options.Hooks);
        var results = new LayaResult?[reqs.Count];
        foreach (var (model, indices) in groups)
        {
            var agent = Load(model);
            var started = new List<PredictContext>(indices.Count);
            try
            {
                // One context per request, started as Predict starts one, so a hook can redact,
                // rewrite or skip each request before it joins a shared forward pass.
                foreach (var i in indices)
                {
                    var ctx = new PredictContext([reqs[i].State], reqs[i].Questions)
                    {
                        Decision = decisions[i],
                        Model = model,
                        Agent = agent,
                        Router = this,
                        MaxLen = options.MaxLen,
                        HeadMaxLen = options.HeadMaxLen,
                        Lang = reqs[i].Lang ?? decisions[i].Detection?.Language,
                    };
                    started.Add(ctx);
                    Dispatch(active, h => h.OnPredictStart(ctx), raise, nameof(ILayaHook.OnPredictStart));
                }

                foreach (var pass in PassGroups(indices, started, decisions, agent))
                {
                    var batch = agent.PredictBatch(pass.Items.Select(x => x.Ctx.States[0]), pass.Questions, new BatchOptions
                    {
                        BatchSize = options.BatchSize,
                        SortByLength = options.SortByLength,
                        MaxLen = pass.MaxLen,
                        HeadMaxLen = pass.HeadMaxLen,
                        Lang = pass.Lang,
                    });
                    if (batch.Count != pass.Items.Count)
                        throw new InvalidOperationException($"internal error: Agent.PredictBatch returned {batch.Count} results for {pass.Items.Count} states");
                    for (var k = 0; k < batch.Count; k++)
                        pass.Items[k].Ctx.Results = [batch[k].WithRouting(decisions[pass.Items[k].Index])];
                }
            }
            catch (Exception ex)
            {
                // Every started request is ended, so a hook that opens something in start always
                // sees the matching end. A request that already has its result keeps it.
                foreach (var ctx in started.Where(c => c.Results is null))
                {
                    ctx.Error = ex;
                    try { Dispatch(active, h => h.OnError(ctx), raise, nameof(ILayaHook.OnError)); }
                    catch (Exception hookEx) { Logger.ErrorHookFailed(hookEx, ex.GetType().Name); }
                }
                try { EndContexts(active, started, raise); }
                catch (Exception hookEx) { Logger.EndHookFailed(hookEx, ex.GetType().Name); }
                throw;
            }

            EndContexts(active, started, raise);
            for (var k = 0; k < indices.Count; k++) results[indices[k]] = started[k].Results![0];
        }

        if (results.Any(r => r is null)) throw new InvalidOperationException("internal error: batch execution did not produce every result");
        return results!;
    }

    /// <summary>
    /// Split one checkpoint's started requests into forward-pass groups on what the start hooks left:
    /// the same question set (order-sensitive, since options are positional), token budget and, only
    /// when the agent has per-language temperatures, language. Skipped requests keep their results.
    /// </summary>
    private static List<PassGroup> PassGroups(List<int> indices, List<PredictContext> started, List<RouteDecision> decisions, LayaAgent agent)
    {
        var passes = new List<PassGroup>();
        // Requests usually share one Questions instance; serialize each instance once.
        var schemas = new Dictionary<Questions, string>(ReferenceEqualityComparer.Instance);
        for (var k = 0; k < indices.Count; k++)
        {
            var ctx = started[k];
            if (ctx.Results is not null)
            {
                // A cache hit short-circuits inference; keep the routing Predict would add.
                ctx.Results = ctx.Results.Select(r => r.WithRouting(decisions[indices[k]])).ToList();
                continue;
            }
            var lang = agent.Temperatures.HasLanguageOverrides ? ctx.Lang : null;
            if (!schemas.TryGetValue(ctx.Questions, out var schema))
                schemas[ctx.Questions] = schema = ctx.Questions.ToJson().ToJsonString();
            var pass = passes.Find(p => p.Schema == schema && p.MaxLen == ctx.MaxLen && p.HeadMaxLen == ctx.HeadMaxLen && p.Lang == lang);
            if (pass is null) passes.Add(pass = new PassGroup(schema, ctx.Questions, ctx.MaxLen, ctx.HeadMaxLen, lang));
            pass.Items.Add((indices[k], ctx));
        }
        return passes;
    }

    private sealed record PassGroup(string Schema, Questions Questions, int? MaxLen, int? HeadMaxLen, string? Lang)
    {
        public List<(int Index, PredictContext Ctx)> Items { get; } = [];
    }

    /// <summary>
    /// End each request of a batch as Predict's <c>finally</c> ends one: every context gets its
    /// <c>OnPredictEnd</c> even if an earlier one's end hook throws; the first such failure on a
    /// context that had not failed is thrown afterwards.
    /// </summary>
    private void EndContexts(ILayaHook[] active, List<PredictContext> contexts, bool raise)
    {
        foreach (var ctx in contexts)
        {
            ctx.ElapsedMs = ctx.ElapsedNow();
            if (ctx.Results is not null) ctx.Usage = Usage.Sum(ctx.Results);
        }
        Exception? first = null;
        foreach (var ctx in contexts)
        {
            try { Dispatch(active, h => h.OnPredictEnd(ctx), raise, nameof(ILayaHook.OnPredictEnd)); }
            catch (Exception hookEx)
            {
                if (ctx.Error is not null) Logger.EndHookFailed(hookEx, ctx.Error.GetType().Name);
                else first ??= hookEx;
            }
        }
        if (first is not null) ExceptionDispatchInfo.Throw(first);
    }

    /// <summary>The same questions over many states, each routed on its own; see <see cref="PredictBatch(IEnumerable{RouteRequest}, BatchOptions?)"/>.</summary>
    public IReadOnlyList<LayaResult> PredictBatch(IEnumerable<LayaState> states, Questions questions, BatchOptions? options = null) =>
        PredictBatch(states.Select(s => new RouteRequest(s, questions) { Lang = options?.Lang }), options);

    public Task<IReadOnlyList<LayaResult>> PredictBatchAsync(IEnumerable<RouteRequest> requests, BatchOptions? options = null, CancellationToken ct = default) =>
        System.Threading.Tasks.Task.Run(() => PredictBatch(requests, options), ct);

    public Task<IReadOnlyList<LayaResult>> PredictBatchAsync(IEnumerable<LayaState> states, Questions questions, BatchOptions? options = null, CancellationToken ct = default) =>
        System.Threading.Tasks.Task.Run(() => PredictBatch(states, questions, options), ct);

    /// <summary>
    /// Stream heterogeneous requests through <see cref="PredictBatch(IEnumerable{RouteRequest}, BatchOptions?)"/>
    /// a chunk of <see cref="BatchOptions.BatchSize"/> (default <see cref="BatchOptions.DefaultStreamBatchSize"/>)
    /// at a time, yielding results in input order. Each chunk loads the checkpoints it needs, so with
    /// mixed languages keep <see cref="RouterOptions.MaxLoaded"/> at least the number of checkpoints in use.
    /// </summary>
    public IAsyncEnumerable<LayaResult> PredictStreamAsync(IEnumerable<RouteRequest> requests, BatchOptions? options = null, CancellationToken ct = default) =>
        PredictStreamAsync(requests.ToAsyncEnumerable(), options, ct);

    /// <inheritdoc cref="PredictStreamAsync(IEnumerable{RouteRequest}, BatchOptions?, CancellationToken)"/>
    public async IAsyncEnumerable<LayaResult> PredictStreamAsync(IAsyncEnumerable<RouteRequest> requests, BatchOptions? options = null,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(requests);
        var size = options?.BatchSize is > 0 ? options.BatchSize.Value : BatchOptions.DefaultStreamBatchSize;
        await foreach (var chunk in requests.Chunk(size).WithCancellation(ct).ConfigureAwait(false))
        {
            ct.ThrowIfCancellationRequested();
            foreach (var r in await PredictBatchAsync(chunk, options, ct).ConfigureAwait(false))
                yield return r;
        }
    }

    LayaResult ILayaPredictor.Predict(LayaState state, Questions questions, PredictOptions? options) =>
        Predict(state, questions, AsRouteOptions(options));

    Task<LayaResult> ILayaPredictor.PredictAsync(LayaState state, Questions questions, PredictOptions? options, CancellationToken ct) =>
        PredictAsync(state, questions, AsRouteOptions(options), ct);

    /// <summary>Plain <see cref="PredictOptions"/> route by detection, as if no routing option was given.</summary>
    private static RouteOptions? AsRouteOptions(PredictOptions? options) => options switch
    {
        null => null,
        RouteOptions r => r,
        _ => new RouteOptions
        {
            Lang = options.Lang,
            MaxLen = options.MaxLen,
            HeadMaxLen = options.HeadMaxLen,
            Hooks = options.Hooks,
            HooksRaise = options.HooksRaise,
        },
    };

    /// <summary>Disposes the agents this router loaded (attached agents belong to the caller).</summary>
    public void Dispose()
    {
        List<LayaAgent> owned;
        lock (_lock)
        {
            owned = _agents.Where(kv => !_attached.Contains(kv.Key)).Select(kv => kv.Value).ToList();
            _agents.Clear();
            _order.Clear();
        }
        foreach (var a in owned) a.Dispose();
    }
}
