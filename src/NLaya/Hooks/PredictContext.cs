using System.Diagnostics;

using NLaya.Routing;

namespace NLaya;

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
    /// <summary>Wall time of the call, set before end hooks run.</summary>
    public TimeSpan? Elapsed { get; set; }
    public Exception? Error { get; set; }

    internal TimeSpan ElapsedNow() => Stopwatch.GetElapsedTime(_started);

    /// <summary>From a start hook: use these results and skip inference. End hooks still run.</summary>
    public void Skip(IList<LayaResult> results) => Results = results;
}
