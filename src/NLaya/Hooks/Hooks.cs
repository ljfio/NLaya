using System.Diagnostics;
using Microsoft.Extensions.Logging;
using NLaya.Routing;

namespace NLaya;

/// <summary>
/// Opt-in lifecycle hooks (port of <c>laya.hooks</c>). Implement any subset: every method has
/// an empty default. Start hooks may rewrite <see cref="PredictContext.States"/> /
/// <see cref="PredictContext.Questions"/> or call <see cref="PredictContext.Skip"/>; end hooks
/// may rewrite <see cref="PredictContext.Results"/>.
/// </summary>
public interface ILayaHook
{
    void OnPredictStart(PredictContext ctx) { }
    void OnPredictEnd(PredictContext ctx) { }
    /// <summary>Router only: may replace <see cref="PredictContext.Decision"/>.</summary>
    void OnRoute(PredictContext ctx) { }
    /// <summary>Router only: a checkpoint was loaded (names in <see cref="PredictContext.Models"/>).</summary>
    void OnLoad(PredictContext ctx) { }
    /// <summary>Router only: checkpoints were evicted (names in <see cref="PredictContext.Models"/>).</summary>
    void OnEvict(PredictContext ctx) { }
    void OnError(PredictContext ctx) { }
}

/// <summary>Base class for hooks that override only some events (Python <c>BaseHook</c>).</summary>
public abstract class LayaHook : ILayaHook
{
    public virtual void OnPredictStart(PredictContext ctx) { }
    public virtual void OnPredictEnd(PredictContext ctx) { }
    public virtual void OnRoute(PredictContext ctx) { }
    public virtual void OnLoad(PredictContext ctx) { }
    public virtual void OnEvict(PredictContext ctx) { }
    public virtual void OnError(PredictContext ctx) { }

    /// <summary>A hook that runs <paramref name="fn"/> before inference.</summary>
    public static ILayaHook OnStart(Action<PredictContext> fn) => new Start(fn);

    /// <summary>A hook that runs <paramref name="fn"/> after inference.</summary>
    public static ILayaHook OnEnd(Action<PredictContext> fn) => new End(fn);

    private sealed class Start(Action<PredictContext> fn) : LayaHook
    {
        public override void OnPredictStart(PredictContext ctx) => fn(ctx);
    }

    private sealed class End(Action<PredictContext> fn) : LayaHook
    {
        public override void OnPredictEnd(PredictContext ctx) => fn(ctx);
    }
}

/// <summary>Mutable state shared by every hook of one call.</summary>
public sealed class PredictContext
{
    private readonly long _started = Stopwatch.GetTimestamp();

    public PredictContext(IList<LayaState> states, Questions questions)
    {
        States = states;
        Questions = questions;
    }

    public IList<LayaState> States { get; set; }
    public Questions Questions { get; set; }
    /// <summary>Shared by every hook of one call.</summary>
    public string RunId { get; } = Guid.NewGuid().ToString("N");
    public IList<LayaResult>? Results { get; set; }
    /// <summary>Router: the routing decision; an <c>OnRoute</c> hook may replace it.</summary>
    public RouteDecision? Decision { get; set; }
    /// <summary>The resolved checkpoint (model id or router name).</summary>
    public string? Model { get; set; }
    /// <summary>Router lifecycle events: the checkpoint names loaded or evicted.</summary>
    public IReadOnlyList<string>? Models { get; set; }
    public LayaAgent? Agent { get; set; }
    public Router? Router { get; set; }
    /// <summary>Per-call token budget overrides; null uses the checkpoint config.</summary>
    public int? MaxLen { get; set; }
    public int? HeadMaxLen { get; set; }
    public string? Lang { get; set; }
    public Usage? Usage { get; set; }
    public double? ElapsedMs { get; set; }
    public Exception? Error { get; set; }

    internal double ElapsedNow() => Stopwatch.GetElapsedTime(_started).TotalMilliseconds;

    /// <summary>From a start hook: use these results and skip inference. End hooks still run.</summary>
    public void Skip(IList<LayaResult> results) => Results = results;
}

/// <summary>Process-wide default hooks, run before installed and per-call hooks everywhere.</summary>
public static class LayaHooks
{
    private static readonly object Gate = new();
    private static ILayaHook[] _defaults = [];

    public static IReadOnlyList<ILayaHook> Defaults
    {
        get { lock (Gate) return _defaults; }
    }

    public static void SetDefaults(params ILayaHook[] hooks)
    {
        lock (Gate) _defaults = hooks.ToArray();
    }

    public static void AddDefault(ILayaHook hook)
    {
        lock (Gate) _defaults = [.. _defaults, hook];
    }

    public static void ClearDefaults()
    {
        lock (Gate) _defaults = [];
    }
}

/// <summary>A runtime-mutable hook list (Python <c>HookRegistry</c>). Calls read a snapshot.</summary>
public abstract class HookRegistry
{
    private readonly object _mutex = new();
    private ILayaHook[] _hooks = [];
    private readonly object? _dispatchLock;

    protected HookRegistry(IEnumerable<ILayaHook>? hooks, bool hooksRaise, bool hooksConcurrent, ILogger? logger)
    {
        _hooks = hooks?.ToArray() ?? [];
        HooksRaise = hooksRaise;
        _dispatchLock = hooksConcurrent ? null : new object();
        Logger = logger ?? Microsoft.Extensions.Logging.Abstractions.NullLogger.Instance;
    }

    /// <summary>When false, a failing hook is logged as a warning and the call continues.</summary>
    public bool HooksRaise { get; set; }

    protected ILogger Logger { get; }

    public IReadOnlyList<ILayaHook> Hooks => _hooks;

    public HookRegistry AddHook(ILayaHook hook)
    {
        lock (_mutex) _hooks = [.. _hooks, hook];
        return this;
    }

    public bool RemoveHook(ILayaHook hook)
    {
        lock (_mutex)
        {
            var before = _hooks.Length;
            _hooks = _hooks.Where(h => !ReferenceEquals(h, hook)).ToArray();
            return _hooks.Length != before;
        }
    }

    /// <summary>Install hooks until the returned scope is disposed.</summary>
    public IDisposable HooksInstalled(params ILayaHook[] hooks)
    {
        foreach (var h in hooks) AddHook(h);
        return new Scope(this, hooks);
    }

    private sealed class Scope(HookRegistry owner, ILayaHook[] hooks) : IDisposable
    {
        private int _done;

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _done, 1) == 1) return;
            foreach (var h in hooks) owner.RemoveHook(h);
        }
    }

    /// <summary>Defaults, then installed, then per-call hooks.</summary>
    internal ILayaHook[] Compose(IEnumerable<ILayaHook>? perCall) =>
        [.. LayaHooks.Defaults, .. _hooks, .. perCall ?? []];

    internal void Dispatch(IReadOnlyList<ILayaHook> hooks, HookEvent evt, PredictContext ctx, bool raise)
    {
        foreach (var hook in hooks)
        {
            try
            {
                if (_dispatchLock is null) Invoke(hook, evt, ctx);
                else lock (_dispatchLock) Invoke(hook, evt, ctx);
            }
            catch (Exception ex) when (!raise)
            {
                Logger.LogWarning(ex, "laya: hook {Hook}.{Event} failed: {Message}", hook.GetType().Name, evt, ex.Message);
            }
        }
    }

    private static void Invoke(ILayaHook hook, HookEvent evt, PredictContext ctx)
    {
        switch (evt)
        {
            case HookEvent.PredictStart: hook.OnPredictStart(ctx); break;
            case HookEvent.PredictEnd: hook.OnPredictEnd(ctx); break;
            case HookEvent.Route: hook.OnRoute(ctx); break;
            case HookEvent.Load: hook.OnLoad(ctx); break;
            case HookEvent.Evict: hook.OnEvict(ctx); break;
            case HookEvent.Error: hook.OnError(ctx); break;
        }
    }

    /// <summary>
    /// Run <paramref name="body"/> between start and end hooks with Python's failure semantics:
    /// error hooks see the exception, a failing error or end hook never masks it, and end hooks
    /// run on the failure path too.
    /// </summary>
    internal IList<LayaResult> RunWithHooks(ILayaHook[] active, PredictContext ctx, bool raise, Func<PredictContext, IList<LayaResult>> body)
    {
        try
        {
            Dispatch(active, HookEvent.PredictStart, ctx, raise);
            ctx.Results ??= body(ctx);
        }
        catch (Exception ex)
        {
            ctx.Error = ex;
            try { Dispatch(active, HookEvent.Error, ctx, raise); }
            catch (Exception hookEx) { Logger.LogWarning(hookEx, "laya: an error hook failed while handling {Error}", ex.GetType().Name); }
            throw;
        }
        finally
        {
            ctx.ElapsedMs = ctx.ElapsedNow();
            if (ctx.Results is not null) ctx.Usage = Usage.Sum(ctx.Results);
            try
            {
                Dispatch(active, HookEvent.PredictEnd, ctx, raise);
            }
            catch (Exception hookEx) when (ctx.Error is not null)
            {
                Logger.LogWarning(hookEx, "laya: an end hook failed while handling {Error}", ctx.Error.GetType().Name);
            }
        }
        return ctx.Results!;
    }
}

internal enum HookEvent
{
    PredictStart,
    PredictEnd,
    Route,
    Load,
    Evict,
    Error,
}
