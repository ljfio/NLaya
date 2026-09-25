namespace NLaya.Calibration;

/// <summary>
/// Accuracy and calibration of a set of answers, the metric block of laya's benchmark harness
/// (<c>hard_metrics</c>), so NLaya's numbers can be compared with BENCHMARKS.md.
/// </summary>
public sealed record ClassificationReport
{
    /// <summary>Answers scored.</summary>
    public int Count { get; init; }

    /// <summary>Share of answers whose most probable option is the correct one.</summary>
    public double Accuracy { get; init; }

    /// <summary>F1 per class (over the classes seen as gold or predicted), averaged.</summary>
    public double MacroF1 { get; init; }

    /// <summary>Expected calibration error of max(p) over 15 equal-width bins; lower is better.</summary>
    public double Ece { get; init; }

    /// <summary>Mean squared distance between the probabilities and the one-hot answer; lower is better.</summary>
    public double Brier { get; init; }

    /// <summary>Mean negative log probability of the correct option; lower is better.</summary>
    public double Nll { get; init; }

    /// <summary>Area under the risk-coverage curve (error rate as answers are admitted most confident first); lower is better.</summary>
    public double Aurc { get; init; }

    /// <summary>Mean max(p).</summary>
    public double MeanConfidence { get; init; }

    /// <summary>Accuracy on the most confident half of the answers.</summary>
    public double AccuracyAt50Coverage { get; init; }

    /// <summary>Accuracy on the most confident 80% of the answers.</summary>
    public double AccuracyAt80Coverage { get; init; }
}
