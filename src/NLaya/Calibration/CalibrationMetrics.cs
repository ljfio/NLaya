namespace NLaya.Calibration;

/// <summary>
/// The calibration and accuracy metrics laya reports: <c>ece_score</c> from <c>laya.common</c> and
/// the benchmark harness's <c>hard_metrics</c> and <c>softmax_t</c>, computed the same way
/// (float64, equal-width bins, first bin closed at 0).
/// </summary>
public static class CalibrationMetrics
{
    /// <summary>The number of confidence bins laya's ECE uses.</summary>
    public const int DefaultBins = 15;

    /// <summary>
    /// Expected calibration error: over equal-width confidence bins, the gap between mean confidence
    /// and accuracy, weighted by the bin's share. NaN for no answers.
    /// </summary>
    public static double Ece(IReadOnlyList<double> confidence, IReadOnlyList<bool> correct, int bins = DefaultBins)
    {
        ArgumentNullException.ThrowIfNull(confidence);
        ArgumentNullException.ThrowIfNull(correct);
        if (confidence.Count != correct.Count) throw new ArgumentException("confidence and correct differ in length", nameof(correct));
        ArgumentOutOfRangeException.ThrowIfLessThan(bins, 1);
        var n = confidence.Count;
        if (n == 0) return double.NaN;
        var e = 0.0;
        for (var b = 0; b < bins; b++)
        {
            // np.linspace(0, 1, bins + 1) edges; the first bin also takes confidence == 0.
            var lo = (double)b / bins;
            var hi = b == bins - 1 ? 1.0 : (double)(b + 1) / bins;
            double count = 0, conf = 0, acc = 0;
            for (var i = 0; i < n; i++)
            {
                var c = confidence[i];
                if ((b == 0 ? c >= lo : c > lo) && c <= hi)
                {
                    count++;
                    conf += c;
                    acc += correct[i] ? 1 : 0;
                }
            }
            if (count > 0) e += count / n * Math.Abs(conf / count - acc / count);
        }
        return e;
    }

    /// <summary><c>softmax(logits / t)</c> in float64, with t floored at 1e-3 (the harness's <c>softmax_t</c>).</summary>
    public static double[] Softmax(IReadOnlyList<float> logits, double temperature = 1.0)
    {
        ArgumentNullException.ThrowIfNull(logits);
        var t = Math.Max(1e-3, temperature);
        var z = logits.Select(x => x / t).ToArray();
        var max = z.Max();
        var e = z.Select(x => Math.Exp(x - max)).ToArray();
        var sum = e.Sum();
        return e.Select(x => x / sum).ToArray();
    }

    /// <summary>Accuracy, macro-F1, ECE, Brier, NLL, AURC and coverage accuracies of <paramref name="predictions"/>.</summary>
    public static ClassificationReport Evaluate(IEnumerable<LabelledPrediction> predictions)
    {
        ArgumentNullException.ThrowIfNull(predictions);
        var rows = predictions.ToList();
        var n = rows.Count;
        if (n == 0) return new ClassificationReport();

        var gold = rows.Select(r => r.Gold).ToArray();
        var pred = rows.Select(r => ArgMax(r.Probabilities)).ToArray();
        var conf = rows.Select(r => r.Probabilities.Max()).ToArray();
        var correct = gold.Zip(pred, (g, p) => g == p).ToArray();
        // Most confident first; stable, so ties keep input order.
        var order = Enumerable.Range(0, n).OrderByDescending(i => conf[i]).ToArray();

        double brier = 0, nll = 0;
        foreach (var r in rows)
        {
            for (var k = 0; k < r.Probabilities.Count; k++)
            {
                var d = r.Probabilities[k] - (k == r.Gold ? 1 : 0);
                brier += d * d;
            }
            nll -= Math.Log(Math.Max(r.Probabilities[r.Gold], 1e-12));
        }

        double wrong = 0, aurc = 0;
        for (var i = 0; i < n; i++)
        {
            wrong += correct[order[i]] ? 0 : 1;
            aurc += wrong / (i + 1);
        }

        double AtCoverage(double coverage)
        {
            var k = Math.Max(1, (int)(n * coverage));
            return order.Take(k).Count(i => correct[i]) / (double)k;
        }

        return new ClassificationReport
        {
            Count = n,
            Accuracy = correct.Count(c => c) / (double)n,
            MacroF1 = MacroF1(gold, pred),
            Ece = Ece(conf, correct),
            Brier = brier / n,
            Nll = nll / n,
            Aurc = aurc / n,
            MeanConfidence = conf.Average(),
            AccuracyAt50Coverage = AtCoverage(0.5),
            AccuracyAt80Coverage = AtCoverage(0.8),
        };
    }

    private static int ArgMax(IReadOnlyList<double> p)
    {
        var best = 0;
        for (var i = 1; i < p.Count; i++) if (p[i] > p[best]) best = i;
        return best;
    }

    private static double MacroF1(int[] gold, int[] pred)
    {
        var classes = gold.Union(pred).Order();
        return classes.Select(c =>
        {
            int tp = 0, fp = 0, fn = 0;
            for (var i = 0; i < gold.Length; i++)
            {
                if (pred[i] == c && gold[i] == c) tp++;
                else if (pred[i] == c) fp++;
                else if (gold[i] == c) fn++;
            }
            return 2.0 * tp / Math.Max(1, 2 * tp + fp + fn);
        }).Average();
    }
}
