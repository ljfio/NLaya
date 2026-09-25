namespace NLaya.Calibration;

/// <summary>One scored answer for <see cref="CalibrationMetrics.Evaluate"/>: the option probabilities and the index of the correct option.</summary>
public sealed record LabelledPrediction(int Gold, IReadOnlyList<double> Probabilities);
