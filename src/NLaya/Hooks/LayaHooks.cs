namespace NLaya;

/// <summary>Hook helpers and the process-wide defaults, which run before installed and per-call hooks.</summary>
public static class LayaHooks
{
    private static ILayaHook[] _defaults = [];

    public static IReadOnlyList<ILayaHook> Defaults => Volatile.Read(ref _defaults);
    public static void SetDefaults(params ILayaHook[] hooks) => Volatile.Write(ref _defaults, hooks.ToArray());
    public static void AddDefault(ILayaHook hook) => Update(ref _defaults, h => [.. h, hook]);
    public static void ClearDefaults() => Volatile.Write(ref _defaults, []);

    /// <summary>A hook that runs <paramref name="fn"/> before inference.</summary>
    public static ILayaHook OnStart(Action<PredictContext> fn) => new DelegateHook(fn, null);

    /// <summary>A hook that runs <paramref name="fn"/> after inference.</summary>
    public static ILayaHook OnEnd(Action<PredictContext> fn) => new DelegateHook(null, fn);

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
