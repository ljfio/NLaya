using NLaya.Calibration;

namespace NLaya.Eval;

/// <summary>One answered question: its raw logits and what it should have answered.</summary>
internal sealed record ScoredQuestion(int Case, string Id, QuestionType Type, float[] Logits, Gold Gold, string? Workflow)
{
    public string Bucket => TemperatureTable.Bucket(Type, Logits.Length);

    public LabelledPrediction At(double temperature) => new(Gold.Index, CalibrationMetrics.Softmax(Logits, temperature));

    /// <summary>A fitting sample: the teacher distribution when there is one (as the notebook fits), else the label.</summary>
    public CalibrationSample Sample(bool soft) => soft && Gold.Soft is { } s && s.Length == Logits.Length
        ? new CalibrationSample(Type, Logits, s)
        : CalibrationSample.FromLabel(Type, Logits, Gold.Index);
}
