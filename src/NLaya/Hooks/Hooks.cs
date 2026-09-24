using System.Diagnostics;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using NLaya.Routing;

namespace NLaya;

/// <summary>
/// Opt-in lifecycle hooks (port of <c>laya.hooks</c>). Implement only the events you need; the rest
/// default to no-ops. Start hooks may rewrite <see cref="PredictContext.States"/> /
/// <see cref="PredictContext.Questions"/> or call <see cref="PredictContext.Skip"/>; end hooks may
/// rewrite <see cref="PredictContext.Results"/>.
/// </summary>
public interface ILayaHook
{
    void OnPredictStart(PredictContext ctx) { }
    void OnPredictEnd(PredictContext ctx) { }
    /// <summary>Router: may replace <see cref="PredictContext.Decision"/>.</summary>
    void OnRoute(PredictContext ctx) { }
    /// <summary>Router: checkpoint <see cref="PredictContext.Model"/> was loaded.</summary>
    void OnLoad(PredictContext ctx) { }
    /// <summary>Router: checkpoint <see cref="PredictContext.Model"/> was evicted or unloaded.</summary>
    void OnEvict(PredictContext ctx) { }
    void OnError(PredictContext ctx) { }
}

/// <summary>Mutable state shared by every hook of one call.</summary>
public sealed class PredictContext(IList<LayaState> states, Questions questions)
{
    private readonly long _started = Stopwatch.GetTimestamp();

    public IList<LayaState> States { get; set; } = states;
    public Questions Questions { get; set; } = questions;
    public string RunId { get; } = Guid.NewGuid().ToString("N");
    public IList<LayaResult>? Results { get; set; }
    public RouteDecision? Decision { get; set; }
    /// <summary>The checkpoint: model id for an agent, checkpoint name for a router.</summary>
    public string? Model { get; set; }
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

/// <summary>Hook helpers and the process-wide defaults, which run before installed and per-call hooks.</summary>
public static class LayaHooks
{
    private static ILayaHook[] _defaults = [];

    public static IReadOnlyList<ILayaHook> Defaults => Volatile.Read(ref _defaults);
    public static void SetDefaults(params ILayaHook[] hooks) => Volatile.Write(ref _defaults, hooks.ToArray());
    public static void AddDefault(ILayaHook hook) => Update(ref _defaults, h => [.. h, hook]);
    public static void ClearDefaults() => Volatile.Write(ref _defaults, []);

    /// <summary>A hook that runs <paramref name="fn"/> before inference.</summary>
    public static ILayaHook OnStart(Action<PredictContext> fn) => new Delegates(fn, null);

    /// <summary>A hook that runs <paramref name="fn"/> after inference.</summary>
    public static ILayaHook OnEnd(Action<PredictContext> fn) => new Delegates(null, fn);

    private sealed class Delegates(Action<PredictContext>? start, Action<PredictContext>? end) : ILayaHook
    {
        public void OnPredictStart(PredictContext ctx) => start?.Invoke(ctx);
        public void OnPredictEnd(PredictContext ctx) => end?.Invoke(ctx);
    }

    internal static void Update(ref ILayaHook[] field, Func<ILayaHook[], ILayaHook[]> change)
    {
        ILayaHook[] seen, next;
        do
        {
            seen = Volatile.Read(ref field);
            next = change(seen);
        } while (Interlocked.CompareExchange(ref field, next, seen) != seen);
    }
}

/// <summary>A hook list that can change at runtime (Python <c>HookRegistry</c>); each call reads a snapshot.</summary>
public abstract class HookRegistry(IEnumerable<ILayaHook>? hooks, bool hooksRaise, ILogger? logger)
{
    private ILayaHook[] _hooks = hooks?.ToArray() ?? [];

    /// <summary>When false, a failing hook is logged as a warning and the call continues.</summary>
    public bool HooksRaise { get; set; } = hooksRaise;

    protected ILogger Logger { get; } = logger ?? NullLogger.Instance;

    public IReadOnlyList<ILayaHook> Hooks => Volatile.Read(ref _hooks);

    public void AddHook(ILayaHook hook) => LayaHooks.Update(ref _hooks, h => [.. h, hook]);

    public bool RemoveHook(ILayaHook hook)
    {
        var removed = false;
        LayaHooks.Update(ref _hooks, h =>
        {
            removed = h.Contains(hook);
            return h.Where(x => !ReferenceEquals(x, hook)).ToArray();
        });
        return removed;
    }

    internal ILayaHook[] Compose(IEnumerable<ILayaHook>? perCall) => [.. LayaHooks.Defaults, .. Hooks, .. perCall ?? []];

    internal void Dispatch(IEnumerable<ILayaHook> hooks, Action<ILayaHook> evt, PredictContext ctx, bool raise, string name)
    {
        foreach (var hook in hooks)
        {
            try { evt(hook); }
            catch (Exception ex) when (!raise)
            {
                Logger.LogWarning(ex, "laya: hook {Hook}.{Event} failed: {Message}", hook.GetType().Name, name, ex.Message);
            }
        }
    }

    /// <summary>
    /// Start hooks, <paramref name="body"/> (unless a start hook skipped it), then end hooks, with
    /// Python's failure rules: error hooks see the exception, and a failing error or end hook never
    /// masks the original one.
    /// </summary>
    internal IList<LayaResult> RunWithHooks(ILayaHook[] active, PredictContext ctx, bool raise, Func<PredictContext, IList<LayaResult>> body)
    {
        try
        {
            Dispatch(active, h => h.OnPredictStart(ctx), ctx, raise, nameof(ILayaHook.OnPredictStart));
            ctx.Results ??= body(ctx);
        }
        catch (Exception ex)
        {
            ctx.Error = ex;
            try { Dispatch(active, h => h.OnError(ctx), ctx, raise, nameof(ILayaHook.OnError)); }
            catch (Exception hookEx) { Logger.LogWarning(hookEx, "laya: an error hook failed while handling {Error}", ex.GetType().Name); }
            throw;
        }
        finally
        {
            ctx.ElapsedMs = ctx.ElapsedNow();
            if (ctx.Results is not null) ctx.Usage = Usage.Sum(ctx.Results);
            try { Dispatch(active, h => h.OnPredictEnd(ctx), ctx, raise, nameof(ILayaHook.OnPredictEnd)); }
            catch (Exception hookEx) when (ctx.Error is not null)
            {
                Logger.LogWarning(hookEx, "laya: an end hook failed while handling {Error}", ctx.Error.GetType().Name);
            }
        }
        return ctx.Results!;
    }
}
