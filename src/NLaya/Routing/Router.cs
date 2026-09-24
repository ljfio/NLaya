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
        ["en"] = "english", ["laya"] = "english", ["default"] = "english",
        ["multi"] = "multilingual", ["ml"] = "multilingual", ["laya-multilingual"] = "multilingual",
        ["typed"] = "typed-decisions", ["typed_decisions"] = "typed-decisions",
        ["laya-typed-decisions"] = "typed-decisions", ["decisions"] = "typed-decisions",
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
            Dispatch(Compose(null), evict ? h => h.OnEvict(ctx) : h => h.OnLoad(ctx), ctx, HooksRaise, evict ? "OnEvict" : "OnLoad");
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
        Dispatch(Compose(options.Hooks), h => h.OnRoute(ctx), ctx, options.HooksRaise ?? HooksRaise, nameof(ILayaHook.OnRoute));
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

    public LayaResult Predict(object state, Questions questions, RouteOptions? options = null) =>
        Predict(LayaState.From(state), questions, options);

    public Task<LayaResult> PredictAsync(LayaState state, Questions questions, RouteOptions? options = null, CancellationToken ct = default) =>
        System.Threading.Tasks.Task.Run(() => Predict(state, questions, options), ct);

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
