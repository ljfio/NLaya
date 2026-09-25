namespace NLaya.Calibration;

/// <summary>
/// One answered question for temperature fitting: its raw logits (from <see cref="LayaAgent.PredictLogits"/>)
/// and the target distribution over its options. A labelled answer's target is one-hot
/// (<see cref="FromLabel"/>); soft targets, such as the typed-decisions dataset's teacher
/// probabilities, are used as given.
/// </summary>
public sealed record CalibrationSample
{
    /// <summary>A sample with a target distribution; <paramref name="target"/> has one entry per logit.</summary>
    public CalibrationSample(QuestionType type, IReadOnlyList<float> logits, IReadOnlyList<float> target)
    {
        ArgumentNullException.ThrowIfNull(logits);
        ArgumentNullException.ThrowIfNull(target);
        if (logits.Count == 0) throw new ArgumentException("a sample needs at least one logit", nameof(logits));
        if (target.Count != logits.Count) throw new ArgumentException($"target has {target.Count} entries for {logits.Count} logits", nameof(target));
        Type = type;
        Logits = logits;
        Target = target;
    }

    /// <summary>The question type; temperatures are fitted per type, or per type and option count.</summary>
    public QuestionType Type { get; }

    /// <summary>Raw option logits, before temperature.</summary>
    public IReadOnlyList<float> Logits { get; }

    /// <summary>The target distribution over the options.</summary>
    public IReadOnlyList<float> Target { get; }

    /// <summary>The option-count bucket this sample calibrates, e.g. "choice:3-5".</summary>
    public string Bucket => TemperatureTable.Bucket(Type, Logits.Count);

    /// <summary>A sample whose correct option is <paramref name="gold"/> (for a noul, 1 is true).</summary>
    public static CalibrationSample FromLabel(QuestionType type, IReadOnlyList<float> logits, int gold)
    {
        ArgumentNullException.ThrowIfNull(logits);
        ArgumentOutOfRangeException.ThrowIfNegative(gold);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(gold, logits.Count);
        var target = new float[logits.Count];
        target[gold] = 1;
        return new CalibrationSample(type, logits, target);
    }

    /// <summary>The index of the largest target, i.e. the label of a one-hot sample.</summary>
    public int Gold
    {
        get
        {
            var best = 0;
            for (var i = 1; i < Target.Count; i++) if (Target[i] > Target[best]) best = i;
            return best;
        }
    }
}
