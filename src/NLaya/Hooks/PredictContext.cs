using System.Diagnostics;

using NLaya.Routing;

namespace NLaya;

/// <summary>Mutable state shared by every hook of one call.</summary>
public sealed class PredictContext(IList<LayaState> states, Questions questions)
{
    private readonly long _started = Stopwatch.GetTimestamp();

    /// <summary>The states being answered; a start hook may replace them.</summary>
    public IList<LayaState> States { get; set; } = states;
    /// <summary>The questions being asked; a start hook may replace them.</summary>
    public Questions Questions { get; set; } = questions;
    /// <summary>A unique id for this call, for correlating start and end.</summary>
    public string RunId { get; } = Guid.NewGuid().ToString("N");
    /// <summary>The results, set after inference (or by <see cref="Skip"/>); an end hook may replace them.</summary>
    public IList<LayaResult>? Results { get; set; }
    /// <summary>Router calls: the routing decision.</summary>
    public RouteDecision? Decision { get; set; }
    /// <summary>The checkpoint: model id for an agent, checkpoint name for a router.</summary>
    public string? Model { get; set; }
    /// <summary>The agent answering (for a router, the chosen checkpoint's).</summary>
    public LayaAgent? Agent { get; set; }
    /// <summary>The router, for router calls.</summary>
    public Router? Router { get; set; }
    /// <summary>Per-call token budget overrides; null uses the checkpoint config.</summary>
    public int? MaxLen { get; set; }
    /// <summary>Per-call question budget; null uses the checkpoint config.</summary>
    public int? HeadMaxLen { get; set; }
    /// <summary>The request's language code, which selects per-language temperatures.</summary>
    public string? Lang { get; set; }
    /// <summary>Tokens read, set before end hooks run.</summary>
    public Usage? Usage { get; set; }
    /// <summary>Wall time of the call, set before end hooks run.</summary>
    public TimeSpan? Elapsed { get; set; }
    /// <summary>The failure, when inference threw.</summary>
    public Exception? Error { get; set; }

    internal TimeSpan ElapsedNow() => Stopwatch.GetElapsedTime(_started);

    /// <summary>From a start hook: use these results and skip inference. End hooks still run.</summary>
    public void Skip(IList<LayaResult> results) => Results = results;
}
