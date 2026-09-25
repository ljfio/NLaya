namespace NLaya.Calibration;

/// <summary>How <see cref="TemperatureFitter"/> searches for the temperature that minimises cross-entropy.</summary>
public enum TemperatureFitMethod
{
    /// <summary>
    /// What the fine-tuning notebook's <c>fit_one_temp</c> aims for: the minimum over T in [0.1, 10].
    /// Needs 10 samples, else 1.0. This finds the minimum exactly; the notebook's LBFGS (fixed step, no
    /// line search) usually lands within 0.2% of it but can stop short, so its temperature is never better.
    /// </summary>
    Optimize,

    /// <summary>
    /// The benchmark harness's <c>fit_temperature</c>: the best of 160 geometrically spaced values in
    /// [0.2, 10], rounded to 4 places. Needs 25 samples, else 1.0. Use it to reproduce laya's published
    /// calibration numbers.
    /// </summary>
    Grid,
}
