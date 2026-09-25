namespace NLaya.Calibration;

/// <summary>
/// Fits calibration temperatures on held-out answers, as laya does after training: one temperature
/// per question type (the fine-tuning notebook) or per type and option-count bucket (the benchmark
/// harness). Each temperature minimises the cross-entropy of <c>softmax(logits / T)</c> against the
/// samples' targets. Temperature doesn't change which option wins, only how confident the answer is,
/// so it moves calibration (ECE, Brier), never accuracy.
/// </summary>
/// <example>
/// <code>
/// var logits = agent.PredictLogits(heldOut.Select(c => c.State), questions);
/// var samples = heldOut.Select((c, i) => CalibrationSample.FromLabel(QuestionType.Choice, logits[i]["intent"], c.GoldIndex));
/// var fit = TemperatureFitter.FitPerType(samples);
/// fit.SaveConfig(Path.Combine(checkpoint, "rl_agent_config.json"), Path.Combine(myCopy, "rl_agent_config.json"));
/// </code>
/// </example>
public static class TemperatureFitter
{
    /// <summary>Fewest samples <see cref="TemperatureFitMethod.Optimize"/> fits from (the notebook's threshold).</summary>
    public const int MinOptimizeSamples = 10;

    /// <summary>Fewest samples <see cref="TemperatureFitMethod.Grid"/> fits from, and a bucket needs in <see cref="FitPerBucket"/> (the harness's threshold).</summary>
    public const int MinGridSamples = 25;

    /// <summary>One temperature for all of <paramref name="samples"/>; 1.0 when there are too few to fit.</summary>
    public static double Fit(IEnumerable<CalibrationSample> samples, TemperatureFitMethod method = TemperatureFitMethod.Optimize)
    {
        ArgumentNullException.ThrowIfNull(samples);
        var list = samples.ToList();
        return method switch
        {
            TemperatureFitMethod.Optimize => list.Count < MinOptimizeSamples ? 1.0 : Optimize(list),
            TemperatureFitMethod.Grid => list.Count < MinGridSamples ? 1.0 : Grid(list),
            _ => throw new ArgumentOutOfRangeException(nameof(method)),
        };
    }

    /// <summary>
    /// One temperature per question type, as the fine-tuning notebook fits them. Types without samples
    /// keep <paramref name="current"/> (default 1.0). The fit has no option-count buckets: saved to a
    /// config it removes <c>temperature_by_options</c>, whose inherited values would otherwise override it.
    /// </summary>
    public static TemperatureFit FitPerType(IEnumerable<CalibrationSample> samples, TemperatureFitMethod method = TemperatureFitMethod.Optimize,
        IReadOnlyList<double>? current = null)
    {
        ArgumentNullException.ThrowIfNull(samples);
        var perType = Current(current);
        var counts = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var group in samples.GroupBy(s => s.Type))
        {
            var list = group.ToList();
            perType[(int)group.Key] = Fit(list, method);
            counts[Question.TypeNameOf(group.Key)] = list.Count;
        }
        return new TemperatureFit(perType, new Dictionary<string, double>(StringComparer.Ordinal), counts);
    }

    /// <summary>
    /// One temperature per option-count bucket ("choice:3-5", ...), as the benchmark harness refits
    /// them. Buckets with fewer than <see cref="MinGridSamples"/> samples are left out, so they fall
    /// back to the per-type temperatures, which stay <paramref name="current"/>.
    /// </summary>
    public static TemperatureFit FitPerBucket(IEnumerable<CalibrationSample> samples, TemperatureFitMethod method = TemperatureFitMethod.Grid,
        IReadOnlyList<double>? current = null)
    {
        ArgumentNullException.ThrowIfNull(samples);
        var byOptions = new Dictionary<string, double>(StringComparer.Ordinal);
        var counts = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var group in samples.GroupBy(s => s.Bucket, StringComparer.Ordinal))
        {
            var list = group.ToList();
            counts[group.Key] = list.Count;
            if (list.Count >= MinGridSamples) byOptions[group.Key] = Fit(list, method);
        }
        return new TemperatureFit(Current(current), byOptions, counts);
    }

    /// <summary>Mean cross-entropy of <c>softmax(logits / t)</c> against the targets, in float64.</summary>
    public static double Loss(IReadOnlyList<CalibrationSample> samples, double temperature)
    {
        ArgumentNullException.ThrowIfNull(samples);
        return Total(samples, temperature) / samples.Count;
    }

    private static double[] Current(IReadOnlyList<double>? current)
    {
        if (current is not null && current.Count != 3) throw new ArgumentException("current must have 3 temperatures (choice, score, noul)", nameof(current));
        return current?.ToArray() ?? [1.0, 1.0, 1.0];
    }

    private static double Total(IReadOnlyList<CalibrationSample> samples, double t)
    {
        var total = 0.0;
        foreach (var s in samples)
        {
            var z = s.Logits;
            var max = double.NegativeInfinity;
            for (var i = 0; i < z.Count; i++) max = Math.Max(max, z[i] / t);
            var sum = 0.0;
            for (var i = 0; i < z.Count; i++) sum += Math.Exp(z[i] / t - max);
            var logSum = Math.Log(sum) + max;
            for (var i = 0; i < z.Count; i++)
                if (s.Target[i] != 0) total -= s.Target[i] * (z[i] / t - logSum);
        }
        return total;
    }

    /// <summary>
    /// The notebook minimises over log T with LBFGS, then clamps T to [0.1, 10]. The loss is unimodal in
    /// log T, so a golden-section search over the clamped range finds the same temperature.
    /// </summary>
    private static double Optimize(List<CalibrationSample> samples)
    {
        const double phi = 0.6180339887498949;
        double lo = Math.Log(0.1), hi = Math.Log(10.0);
        double a = hi - phi * (hi - lo), b = lo + phi * (hi - lo);
        double fa = Total(samples, Math.Exp(a)), fb = Total(samples, Math.Exp(b));
        while (hi - lo > 1e-10)
        {
            if (fa <= fb)
            {
                hi = b;
                (b, fb) = (a, fa);
                a = hi - phi * (hi - lo);
                fa = Total(samples, Math.Exp(a));
            }
            else
            {
                lo = a;
                (a, fa) = (b, fb);
                b = lo + phi * (hi - lo);
                fb = Total(samples, Math.Exp(b));
            }
        }
        return Math.Exp((lo + hi) / 2);
    }

    /// <summary>
    /// <c>np.geomspace(0.2, 10.0, 160)</c>, keeping the first temperature with the lowest total loss,
    /// rounded to 4 places as the harness reports it.
    /// </summary>
    private static double Grid(List<CalibrationSample> samples)
    {
        const int steps = 160;
        double logLo = Math.Log10(0.2), logHi = Math.Log10(10.0);
        var step = (logHi - logLo) / (steps - 1);
        double bestT = 1.0, best = double.PositiveInfinity;
        for (var i = 0; i < steps; i++)
        {
            var t = i == 0 ? 0.2 : i == steps - 1 ? 10.0 : Math.Pow(10, logLo + i * step);
            var total = Total(samples, t);
            if (total < best) (best, bestT) = (total, t);
        }
        return Math.Round(bestT, 4, MidpointRounding.ToEven);
    }
}
