using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

namespace NLaya;

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

    internal void Dispatch(IEnumerable<ILayaHook> hooks, Action<ILayaHook> evt, bool raise, string name)
    {
        foreach (var hook in hooks)
        {
            try { evt(hook); }
            catch (Exception ex) when (!raise)
            {
                Logger.HookFailed(ex, hook.GetType().Name, name, ex.Message);
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
            Dispatch(active, h => h.OnPredictStart(ctx), raise, nameof(ILayaHook.OnPredictStart));
            ctx.Results ??= body(ctx);
        }
        catch (Exception ex)
        {
            ctx.Error = ex;
            try { Dispatch(active, h => h.OnError(ctx), raise, nameof(ILayaHook.OnError)); }
            catch (Exception hookEx) { Logger.ErrorHookFailed(hookEx, ex.GetType().Name); }
            throw;
        }
        finally
        {
            ctx.ElapsedMs = ctx.ElapsedNow();
            if (ctx.Results is not null) ctx.Usage = Usage.Sum(ctx.Results);
            try { Dispatch(active, h => h.OnPredictEnd(ctx), raise, nameof(ILayaHook.OnPredictEnd)); }
            catch (Exception hookEx) when (ctx.Error is not null)
            {
                Logger.EndHookFailed(hookEx, ctx.Error.GetType().Name);
            }
        }
        return ctx.Results!;
    }
}
