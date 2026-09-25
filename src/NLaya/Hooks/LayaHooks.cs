namespace NLaya;

/// <summary>Hook helpers and the process-wide defaults, which run before installed and per-call hooks.</summary>
public static class LayaHooks
{
    private static ILayaHook[] s_defaults = [];

    /// <summary>Hooks every agent and router runs, before their own.</summary>
    public static IReadOnlyList<ILayaHook> Defaults => Volatile.Read(ref s_defaults);
    /// <summary>Replace the process-wide hooks. Prefer <c>AddLayaHook</c> with dependency injection in hosted apps.</summary>
    public static void SetDefaults(params ILayaHook[] hooks) => Volatile.Write(ref s_defaults, hooks.ToArray());
    /// <summary>Add a process-wide hook.</summary>
    public static void AddDefault(ILayaHook hook) => Update(ref s_defaults, h => [.. h, hook]);
    /// <summary>Remove every process-wide hook.</summary>
    public static void ClearDefaults() => Volatile.Write(ref s_defaults, []);

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
