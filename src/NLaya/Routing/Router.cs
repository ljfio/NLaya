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
    /// <summary>The Hub repo that bundles all three checkpoints (English at the root).</summary>
    public const string BundleRepo = "convaiinnovations/laya";

    /// <summary>Where checkpoints load from by default: subfolders of <see cref="BundleRepo"/>, as in Python.</summary>
    public static readonly IReadOnlyDictionary<Checkpoint, CheckpointSpec> DefaultModels = new Dictionary<Checkpoint, CheckpointSpec>
    {
        [Checkpoint.English] = new(BundleRepo),
        [Checkpoint.Multilingual] = new(BundleRepo, "multilingual"),
        [Checkpoint.TypedDecisions] = new(BundleRepo, "typed-decisions"),
    };

    /// <summary>The standalone repos, used with <see cref="RouterOptions.StandaloneRepos"/>.</summary>
    public static readonly IReadOnlyDictionary<Checkpoint, CheckpointSpec> StandaloneModels = new Dictionary<Checkpoint, CheckpointSpec>
    {
        [Checkpoint.English] = new("convaiinnovations/laya"),
        [Checkpoint.Multilingual] = new("convaiinnovations/laya-multilingual"),
        [Checkpoint.TypedDecisions] = new("convaiinnovations/laya-typed-decisions"),
    };

    private static readonly (string Name, string[] Ids)[] Workflows =
    [
        ("agent_trace_observability", ["action", "needs_review", "outcome", "risk", "urgency"]),
        ("customer_service", ["action", "category", "churn_risk", "needs_human", "urgency"]),
        ("invoice_processing", ["discrepancy_severity", "disposition", "duplicate", "matches_order", "urgency"]),
        ("security_incidents", ["credential_compromise", "disposition", "severity", "true_positive", "urgency"]),
    ];

    private readonly RouterOptions _o;
    private readonly Dictionary<Checkpoint, CheckpointSpec> _models;
    // A checkpoint is in _agents from the moment its load starts, and in _order once it has loaded.
    private readonly Dictionary<Checkpoint, Lazy<LayaAgent>> _agents = [];
    private readonly HashSet<Checkpoint> _attached = [];
    private readonly List<Checkpoint> _order = []; // least recently used first
    // Router calls in flight per agent. An agent evicted or unloaded while in use is retired and
    // disposed when its last call returns, so its (possibly GPU) memory is freed deterministically.
    private readonly Dictionary<LayaAgent, int> _leases = [];
    private readonly HashSet<LayaAgent> _retired = [];
    private readonly Lock _lock = new();
    private int _maxLoaded;

    /// <summary>A router with <paramref name="options"/>; checkpoints load on first use.</summary>
    public Router(RouterOptions? options = null) : base((options ??= new()).Hooks, options.ThrowOnHookError, options.Logger)
    {
        _o = options;
        _models = new Dictionary<Checkpoint, CheckpointSpec>(options.StandaloneRepos ? StandaloneModels : DefaultModels);
        foreach (var (k, v) in options.Models) _models[k] = v;
        _maxLoaded = Math.Max(1, options.MaxLoaded);
        Default = options.Default;
    }

    /// <summary>A router whose checkpoints all load with <paramref name="configure"/>, e.g. <c>o => o.UseTorchSharp()</c>.</summary>
    public Router(Action<LayaOptions> configure) : this(new RouterOptions { ConfigureAgent = (_, o) => configure(o) }) { }

    /// <summary>The checkpoint used when routing can't tell (no letters, or an undecided Latin-script language).</summary>
    public Checkpoint Default { get; }

    /// <summary>Where each checkpoint loads from: the bundle repo (or standalone repos) plus any <see cref="RouterOptions.Models"/> overrides.</summary>
    public IReadOnlyDictionary<Checkpoint, CheckpointSpec> Models => _models;

    /// <summary>The checkpoints loaded now, least recently used first.</summary>
    public IReadOnlyList<Checkpoint> Loaded
    {
        get { lock (_lock) return _order.ToList(); }
    }

    /// <summary>The typed-decisions workflow whose exact question-id set this is, else null.</summary>
    public static string? MatchTypedDecisionsWorkflow(IEnumerable<string>? questionIds)
    {
        var ids = (questionIds ?? []).ToHashSet();
        return Workflows.FirstOrDefault(w => ids.SetEquals(w.Ids)).Name;
    }

    // ---------------------------------------------------------------- loading

    /// <summary>
    /// The agent for a checkpoint, loading it on first use. Loading takes seconds, so it runs outside
    /// the router's lock: requests for checkpoints already loaded carry on, and concurrent requests for
    /// the same checkpoint share one load.
    /// </summary>
    /// <remarks>
    /// The agent belongs to the router (unless you <see cref="Attach"/>ed it): when it is evicted or
    /// unloaded it is disposed as soon as the router's own calls on it finish. Use it while it is
    /// loaded, or call <see cref="Load"/> again, rather than keeping it.
    /// </remarks>
    public LayaAgent Load(Checkpoint key) => LoadCore(key, lease: false);

    /// <summary>Load (or find) the agent and take a lease on it, so eviction can't dispose it mid-call.</summary>
    private AgentLease Lease(Checkpoint key) => new(this, LoadCore(key, lease: true));

    private LayaAgent LoadCore(Checkpoint key, bool lease)
    {
        Lazy<LayaAgent> entry;
        lock (_lock)
        {
            if (_agents.TryGetValue(key, out var existing) && existing.IsValueCreated)
            {
                Touch(key);
                return Acquire(existing.Value, lease);
            }
            entry = existing ?? (_agents[key] = new Lazy<LayaAgent>(() => LoadAgent(key)));
        }

        LayaAgent agent;
        try
        {
            agent = entry.Value;
        }
        catch
        {
            // Lazy caches the exception; forget the entry so the next call tries again.
            lock (_lock)
            {
                if (_agents.TryGetValue(key, out var current) && current == entry) _agents.Remove(key);
            }
            throw;
        }

        List<Checkpoint> evicted;
        List<LayaAgent> dispose;
        lock (_lock)
        {
            if (!_agents.TryGetValue(key, out var current) || current != entry)
            {
                // Unloaded while loading: this caller gets the agent, and nobody else will. A leased
                // one is disposed when the lease ends; an unleased one belongs to the caller.
                if (lease) _retired.Add(agent);
                return Acquire(agent, lease);
            }
            if (_order.Contains(key))
            {
                // Loaded by a concurrent caller: already counted.
                Touch(key);
                return Acquire(agent, lease);
            }
            _order.Add(key);
            (evicted, dispose) = EvictLocked();
            Acquire(agent, lease);
        }
        DisposeAll(dispose);
        // Lifecycle hooks run outside the lock, so a hook may call back into the router.
        Lifecycle(evicted, evict: true);
        Lifecycle([key], evict: false, agent);
        return agent;
    }

    private LayaAgent Acquire(LayaAgent agent, bool lease)
    {
        if (lease) _leases[agent] = _leases.GetValueOrDefault(agent) + 1;
        return agent;
    }

    private void Release(LayaAgent agent)
    {
        lock (_lock)
        {
            var n = _leases[agent] - 1;
            if (n > 0)
            {
                _leases[agent] = n;
                return;
            }
            _leases.Remove(agent);
            if (!_retired.Remove(agent)) return;
        }
        agent.Dispose();
    }

    /// <summary>A router call's hold on an agent; disposing it releases the hold.</summary>
    private readonly struct AgentLease(Router router, LayaAgent agent) : IDisposable
    {
        public LayaAgent Agent => agent;

        public void Dispose() => router.Release(agent);
    }

    private LayaAgent LoadAgent(Checkpoint key)
    {
        var spec = _models[key];
        return Laya.Load(spec.Repo, o =>
        {
            o.Subfolder = spec.Subfolder;
            o.Logger ??= _o.Logger;
            _o.ConfigureAgent?.Invoke(key, o);
        });
    }

    /// <summary>Register an agent you already built instead of loading a second copy. The router never disposes it.</summary>
    public LayaAgent Attach(Checkpoint key, LayaAgent agent)
    {
        ArgumentNullException.ThrowIfNull(agent);
        List<LayaAgent> dispose;
        lock (_lock)
        {
            // A router-loaded agent this replaces is retired like an evicted one.
            dispose = _agents.TryGetValue(key, out var old) && old.IsValueCreated && !_attached.Contains(key) && old.Value != agent
                ? RetireLocked([old.Value])
                : [];
            var loaded = new Lazy<LayaAgent>(() => agent);
            _ = loaded.Value;
            _agents[key] = loaded;
            _attached.Add(key);
            Touch(key);
            _maxLoaded = Math.Max(_maxLoaded, _agents.Count);
        }
        DisposeAll(dispose);
        return agent;
    }

    /// <summary>Load checkpoints up front (all of them by default) so no request pays for a load.</summary>
    public Router Preload(IEnumerable<Checkpoint>? checkpoints = null)
    {
        var keys = (checkpoints ?? _models.Keys).Distinct().ToList();
        lock (_lock) _maxLoaded = Math.Max(_maxLoaded, keys.Union(_agents.Keys).Count());
        foreach (var k in keys) Load(k);
        return this;
    }

    /// <summary><see cref="Preload"/> on the thread pool.</summary>
    public Task<Router> PreloadAsync(IEnumerable<Checkpoint>? checkpoints = null, CancellationToken ct = default) =>
        Task.Run(() => Preload(checkpoints), ct);

    /// <summary>
    /// Drop one checkpoint, or all. Agents the router loaded are disposed once in-flight calls on
    /// them finish; attached agents are only forgotten.
    /// </summary>
    public void Unload(Checkpoint? checkpoint = null)
    {
        List<Checkpoint> freed;
        List<LayaAgent> dispose;
        lock (_lock)
        {
            var keys = checkpoint is { } key ? [key] : _agents.Keys.ToList();
            // Only loaded checkpoints get OnEvict; one still loading just never joins.
            freed = keys.Where(_order.Contains).ToList();
            var owned = new List<LayaAgent>();
            foreach (var k in keys)
            {
                if (_agents.Remove(k, out var entry) && entry.IsValueCreated && !_attached.Contains(k)) owned.Add(entry.Value);
                _attached.Remove(k);
                _order.Remove(k);
            }
            dispose = RetireLocked(owned);
        }
        DisposeAll(dispose);
        Lifecycle(freed, evict: true);
    }

    private void Touch(Checkpoint key)
    {
        _order.Remove(key);
        _order.Add(key);
    }

    private (List<Checkpoint> Evicted, List<LayaAgent> Dispose) EvictLocked()
    {
        var evicted = new List<Checkpoint>();
        var owned = new List<LayaAgent>();
        while (_order.Count > _maxLoaded)
        {
            var victim = _order[0];
            _order.RemoveAt(0);
            if (!_agents.Remove(victim, out var entry)) continue;
            evicted.Add(victim);
            if (!_attached.Remove(victim) && entry.IsValueCreated) owned.Add(entry.Value);
        }
        return (evicted, RetireLocked(owned));
    }

    /// <summary>Agents no longer served: the idle ones are returned to dispose now, the busy ones wait for their last lease.</summary>
    private List<LayaAgent> RetireLocked(List<LayaAgent> agents)
    {
        var idle = new List<LayaAgent>();
        foreach (var a in agents)
        {
            if (_leases.ContainsKey(a)) _retired.Add(a);
            else idle.Add(a);
        }
        return idle;
    }

    private static void DisposeAll(List<LayaAgent> agents)
    {
        foreach (var a in agents) a.Dispose();
    }

    private void Lifecycle(IEnumerable<Checkpoint> checkpoints, bool evict, LayaAgent? agent = null)
    {
        foreach (var n in checkpoints)
        {
            var ctx = new PredictContext([], new Questions()) { Model = n.Name(), Agent = agent, Router = this };
            Dispatch(Compose(null), evict ? h => h.OnEvict(ctx) : h => h.OnLoad(ctx), ThrowOnHookError, evict ? "OnEvict" : "OnLoad");
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
        Dispatch(Compose(options.Hooks), h => h.OnRoute(ctx), options.ThrowOnHookError ?? ThrowOnHookError, nameof(ILayaHook.OnRoute));
        LayaTelemetry.Routed(ctx.Decision!.Checkpoint.Name());
        return ctx.Decision;
    }

    private RouteDecision Decide(LayaState state, Questions? questions, RouteOptions o)
    {
        RouteDecision To(Checkpoint key, string reason, LanguageDetection? det = null, string? wf = null) =>
            new(key, _models[key].ToString(), reason, det, wf);

        if (o.Checkpoint is { } pinned) return To(pinned, $"explicit model={PyStr.Repr(pinned.Name())}");

        var workflow = MatchTypedDecisionsWorkflow(questions?.Keys);
        if (workflow is not null && _o.AutoTaskDetection)
            return To(Checkpoint.TypedDecisions, $"question ids match the {PyStr.Repr(workflow)} typed-decisions workflow", null, workflow);

        if (o.Lang is not null && EnglishFromCode(o.Lang) is { } byLang)
            return To(byLang ? Checkpoint.English : Checkpoint.Multilingual, $"explicit lang={PyStr.Repr(o.Lang)}", null, workflow);

        foreach (var (source, hint) in new[] { ("lang_guess", o.LangGuess), ("Router(lang_guess=...)", _o.LangGuess) })
        {
            if (hint is not null && EnglishFromCode(hint(state)) is { } en)
                return To(en ? Checkpoint.English : Checkpoint.Multilingual,
                    $"{source}: the caller identified this as {(en ? "English" : "non-English")} text", null, workflow);
        }

        var det = LanguageDetector.Analyse(state);
        Checkpoint model;
        string reason;
        if (det.Script == "unknown")
            (model, reason) = (Default, $"no letters detected in state; using default ({Default.Name()})");
        else if (det.Script != "latin")
            (model, reason) = (Checkpoint.Multilingual,
                $"non-Latin script ({det.Script}, {PyStr.F0(100 * det.NonLatinFraction)}% of letters); the English checkpoint cannot read it");
        else if (!det.IsEnglish)
            (model, reason) = (Checkpoint.Multilingual,
                det.MixedSegment is not null
                    ? $"Latin script, mostly English, but a line or field reads as {PyStr.Repr(det.Language!)} ({PyStr.Repr(PyStr.Take(det.MixedSegment, 60))}); the English checkpoint cannot read it"
                    : det.Language is not null
                        ? $"Latin script but language looks like {PyStr.Repr(det.Language)}, not English"
                        : $"Latin script, language not identified but {PyStr.F0(100 * det.DiacriticRate)}% non-English letters; not safe for the English checkpoint");
        else if (det.LanguageUndecided)
            (model, reason) = (Default, $"Latin script, language not identified and no non-English letters; using default ({Default.Name()})");
        else
            (model, reason) = (Checkpoint.English, "English Latin text");
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
        using var lease = Lease(decision.Checkpoint);
        var agent = lease.Agent;
        var lang = options.Lang ?? decision.Detection?.Language;
        var ctx = new PredictContext([state], questions)
        {
            Decision = decision,
            Model = decision.Checkpoint.Name(),
            Agent = agent,
            Router = this,
            MaxLen = options.MaxLen,
            HeadMaxLen = options.HeadMaxLen,
            Lang = lang,
        };
        var results = RunWithHooks(Compose(options.Hooks), ctx, options.ThrowOnHookError ?? ThrowOnHookError, c =>
            [agent.Predict(c.States[0], c.Questions, new PredictOptions { Lang = c.Lang, MaxLen = c.MaxLen, HeadMaxLen = c.HeadMaxLen })]);
        return results[0].WithRouting(decision);
    }

    /// <summary>Route and answer questions about any object (serialized to JSON with System.Text.Json).</summary>
    [RequiresUnreferencedCode(LayaState.ReflectionMessage)]
    [RequiresDynamicCode(LayaState.ReflectionMessage)]
    public LayaResult Predict(object state, Questions questions, RouteOptions? options = null) =>
        Predict(LayaState.From(state), questions, options);

    /// <summary><see cref="Predict(LayaState, Questions, RouteOptions?)"/> on the thread pool; <paramref name="ct"/> only cancels a call that hasn't started yet.</summary>
    public Task<LayaResult> PredictAsync(LayaState state, Questions questions, RouteOptions? options = null, CancellationToken ct = default) =>
        Task.Run(() => Predict(state, questions, options), ct);

    // ---------------------------------------------------------------- batch

    /// <summary>
    /// Route a heterogeneous batch without loading anything (Python <c>route_batch</c>), so a caller
    /// can inspect or aggregate the decisions first. Decisions keep input order.
    /// </summary>
    public IReadOnlyList<RouteDecision> RouteBatch(IEnumerable<RouteRequest> requests) =>
        RouteBatchCore(requests.ToList(), null, null);

    private List<RouteDecision> RouteBatchCore(List<RouteRequest> requests, IEnumerable<ILayaHook>? hooks, bool? throwOnHookError)
    {
        var decisions = new List<RouteDecision>(requests.Count);
        for (var i = 0; i < requests.Count; i++)
        {
            var r = requests[i] ?? throw new ArgumentException($"request {i} is null", nameof(requests));
            if (r.State is null) throw new ArgumentException($"request {i} is missing required key 'state'", nameof(requests));
            if (r.Questions is null) throw new ArgumentException($"request {i} is missing required key 'questions'", nameof(requests));
            decisions.Add(Route(r.State, r.Questions, r.ToRouteOptions(hooks, throwOnHookError)));
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
        var raise = options.ThrowOnHookError ?? ThrowOnHookError;
        var decisions = RouteBatchCore(reqs, options.Hooks, options.ThrowOnHookError);
        if (decisions.Count == 0) return [];

        // First-appearance order keeps loads deterministic and collapses an interleaved workload to
        // one load per checkpoint for this call.
        var groups = new OrderedDictionary<Checkpoint, List<int>>();
        for (var i = 0; i < decisions.Count; i++)
        {
            if (!groups.TryGetValue(decisions[i].Checkpoint, out var indices)) groups[decisions[i].Checkpoint] = indices = [];
            indices.Add(i);
        }

        var active = Compose(options.Hooks);
        var results = new LayaResult?[reqs.Count];
        foreach (var (model, indices) in groups)
        {
            using var lease = Lease(model);
            var agent = lease.Agent;
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
                        Model = model.Name(),
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
            ctx.Elapsed = ctx.ElapsedNow();
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

    /// <summary><see cref="PredictBatch(IEnumerable{RouteRequest}, BatchOptions?)"/> on the thread pool.</summary>
    public Task<IReadOnlyList<LayaResult>> PredictBatchAsync(IEnumerable<RouteRequest> requests, BatchOptions? options = null, CancellationToken ct = default) =>
        Task.Run(() => PredictBatch(requests, options), ct);

    /// <summary><see cref="PredictBatch(IEnumerable{LayaState}, Questions, BatchOptions?)"/> on the thread pool.</summary>
    public Task<IReadOnlyList<LayaResult>> PredictBatchAsync(IEnumerable<LayaState> states, Questions questions, BatchOptions? options = null, CancellationToken ct = default) =>
        Task.Run(() => PredictBatch(states, questions, options), ct);

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
            ThrowOnHookError = options.ThrowOnHookError,
        },
    };

    /// <summary>Disposes the agents this router loaded, including retired ones still in use (attached agents belong to the caller).</summary>
    public void Dispose()
    {
        List<LayaAgent> owned;
        lock (_lock)
        {
            owned = _agents.Where(kv => !_attached.Contains(kv.Key) && kv.Value.IsValueCreated).Select(kv => kv.Value.Value)
                .Concat(_retired).Distinct().ToList();
            _agents.Clear();
            _attached.Clear();
            _order.Clear();
            _retired.Clear();
        }
        DisposeAll(owned);
    }
}
