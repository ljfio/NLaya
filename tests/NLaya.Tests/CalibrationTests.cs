using System.Text.Json.Nodes;

using NLaya.Calibration;

namespace NLaya.Tests;

/// <summary>Temperature fitting and calibration metrics against laya's own code (calibration.json).</summary>
public class CalibrationTests
{
    private static readonly JsonNode Fixture = TestFiles.Fixture("calibration.json");

    public static TheoryData<int> Fits() => [.. Enumerable.Range(0, Fixture["fits"]!.AsArray().Count)];
    public static TheoryData<int> Eces() => [.. Enumerable.Range(0, Fixture["ece"]!.AsArray().Count)];
    public static TheoryData<int> Metrics() => [.. Enumerable.Range(0, Fixture["hard_metrics"]!.AsArray().Count)];

    private static List<CalibrationSample> Samples(JsonNode fit) =>
        fit["samples"]!.AsArray().Select(s => new CalibrationSample(
            s!["type"]!.GetValue<string>() switch { "choice" => QuestionType.Choice, "score" => QuestionType.Score, _ => QuestionType.Noul },
            Floats(s["logits"]!), Floats(s["target"]!))).ToList();

    private static float[] Floats(JsonNode n) => n.AsArray().Select(x => (float)x!.GetValue<double>()).ToArray();

    private static double[] Doubles(JsonNode n) => n.AsArray().Select(x => x!.GetValue<double>()).ToArray();

    [Theory]
    [MemberData(nameof(Fits))]
    public void Optimize_is_at_least_as_good_as_the_notebooks_fit_one_temp(int i)
    {
        var fit = Fixture["fits"]![i]!;
        var samples = Samples(fit);
        var theirs = fit["fit_one_temp"]!.GetValue<double>();
        var ours = TemperatureFitter.Fit(samples, TemperatureFitMethod.Optimize);
        if (samples.Count < TemperatureFitter.MinOptimizeSamples)
        {
            Assert.Equal(theirs, ours);
            return;
        }
        // The notebook's LBFGS (fixed step, no line search) can stop short of the minimum, as it does on
        // "underconfident"; the golden-section search finds it. Where LBFGS converged, the two agree.
        var (lossOurs, lossTheirs) = (TemperatureFitter.Loss(samples, ours), TemperatureFitter.Loss(samples, theirs));
        Assert.True(lossOurs <= lossTheirs + 1e-9, $"T={ours} loss {lossOurs} > notebook T={theirs} loss {lossTheirs}");
        if (lossTheirs - lossOurs < 1e-5) Assert.Equal(theirs, ours, theirs * 2e-3);
    }

    [Theory]
    [MemberData(nameof(Fits))]
    public void Grid_matches_the_harnesss_fit_temperature(int i)
    {
        var fit = Fixture["fits"]![i]!;
        var expected = fit["fit_temperature"];
        if (expected is null) Assert.Skip("the harness fits hard labels only");
        Assert.Equal(expected.GetValue<double>(), TemperatureFitter.Fit(Samples(fit), TemperatureFitMethod.Grid));
    }

    [Theory]
    [MemberData(nameof(Eces))]
    public void Ece_matches_ece_score(int i)
    {
        var c = Fixture["ece"]![i]!;
        var ece = CalibrationMetrics.Ece(Doubles(c["confidence"]!), Doubles(c["correct"]!).Select(x => x > 0).ToArray());
        if (c["ece"] is { } expected) Assert.Equal(expected.GetValue<double>(), ece, 1e-12);
        else Assert.True(double.IsNaN(ece)); // ece_score's NaN, written as null
    }

    [Theory]
    [MemberData(nameof(Metrics))]
    public void Evaluate_matches_hard_metrics(int i)
    {
        var c = Fixture["hard_metrics"]![i]!;
        var report = CalibrationMetrics.Evaluate(c["rows"]!.AsArray()
            .Select(r => new LabelledPrediction(r!["gold"]!.GetValue<int>(), Doubles(r["probabilities"]!))));
        var m = c["metrics"]!;
        double M(string k) => m[k]!.GetValue<double>();
        Assert.Equal(m["n"]!.GetValue<int>(), report.Count);
        Assert.Equal(M("accuracy"), report.Accuracy, 1e-12);
        Assert.Equal(M("macro_f1"), report.MacroF1, 1e-12);
        Assert.Equal(M("ece"), report.Ece, 1e-12);
        Assert.Equal(M("brier"), report.Brier, 1e-12);
        Assert.Equal(M("nll"), report.Nll, 1e-12);
        Assert.Equal(M("aurc"), report.Aurc, 1e-12);
        Assert.Equal(M("mean_confidence"), report.MeanConfidence, 1e-12);
        Assert.Equal(M("acc_at_50_coverage"), report.AccuracyAt50Coverage, 1e-12);
        Assert.Equal(M("acc_at_80_coverage"), report.AccuracyAt80Coverage, 1e-12);
    }

    [Fact]
    public void Softmax_matches_softmax_t()
    {
        foreach (var c in Fixture["softmax_t"]!.AsArray())
        {
            var p = CalibrationMetrics.Softmax(Floats(c!["logits"]!), c["temperature"]!.GetValue<double>());
            var expected = Doubles(c["probs"]!);
            for (var k = 0; k < p.Length; k++) Assert.Equal(expected[k], p[k], 1e-12);
        }
    }

    [Fact]
    public void A_per_type_fit_replaces_temperatures_and_drops_inherited_buckets()
    {
        var samples = Samples(Fixture["fits"]![2]!);
        var fit = TemperatureFitter.FitPerType(samples.Where(s => s.Type != QuestionType.Score), current: [1.1, 1.2, 1.3]);

        Assert.Equal(1.2, fit.PerType[1]); // no score samples: kept
        Assert.Empty(fit.ByOptions);
        var config = JsonNode.Parse("""{"head_layers": 2, "temperature": [1, 1, 1], "temperature_by_options": {"choice:11+": 0.1}}""")!.AsObject();
        var applied = fit.ApplyTo(config);
        Assert.False(applied.ContainsKey("temperature_by_options"));
        Assert.Equal(fit.PerType, applied["temperature"]!.AsArray().Select(t => t!.GetValue<double>()));
        Assert.Equal(2, applied["head_layers"]!.GetValue<int>());
        Assert.True(config.ContainsKey("temperature_by_options")); // the input is not changed
    }

    [Fact]
    public void A_per_bucket_fit_skips_buckets_with_too_few_samples()
    {
        var samples = Samples(Fixture["fits"]![4]!);
        var fit = TemperatureFitter.FitPerBucket(samples);

        foreach (var (bucket, n) in fit.SampleCounts)
            Assert.Equal(n >= TemperatureFitter.MinGridSamples, fit.ByOptions.ContainsKey(bucket));
        Assert.Equal([1.0, 1.0, 1.0], fit.PerType);
        var noul = samples.Where(s => s.Bucket == "noul:2").ToList();
        Assert.Equal(TemperatureFitter.Fit(noul, TemperatureFitMethod.Grid), fit.ByOptions["noul:2"]);
    }

    [Fact]
    public void Temperature_scaling_lowers_ece_on_overconfident_answers()
    {
        var samples = Samples(Fixture["fits"]![2]!);
        var t = TemperatureFitter.Fit(samples);
        ClassificationReport Report(double temperature) => CalibrationMetrics.Evaluate(
            samples.Select(s => new LabelledPrediction(s.Gold, CalibrationMetrics.Softmax(s.Logits, temperature))));

        var raw = Report(1.0);
        var scaled = Report(t);
        Assert.True(t > 1, $"expected an overconfident set to fit T > 1, got {t}");
        Assert.True(scaled.Ece < raw.Ece, $"ECE {raw.Ece} -> {scaled.Ece}");
        Assert.Equal(raw.Accuracy, scaled.Accuracy); // temperature never changes the answer
    }

    [Fact]
    public void SaveConfig_writes_a_copy_and_refuses_links()
    {
        var dir = Directory.CreateTempSubdirectory("nlaya-calibration").FullName;
        try
        {
            var source = Path.Combine(dir, "source.json");
            File.WriteAllText(source, """{"temperature": [1, 1, 1]}""");
            var fit = TemperatureFitter.FitPerType(Samples(Fixture["fits"]![2]!));
            var saved = Path.Combine(dir, "rl_agent_config.json");
            fit.SaveConfig(source, saved);
            Assert.Equal(fit.PerType, JsonNode.Parse(File.ReadAllText(saved))!["temperature"]!.AsArray().Select(t => t!.GetValue<double>()));

            var link = Path.Combine(dir, "link.json");
            File.CreateSymbolicLink(link, source);
            Assert.Throws<IOException>(() => fit.SaveConfig(source, link));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void Samples_are_validated()
    {
        Assert.Throws<ArgumentException>(() => new CalibrationSample(QuestionType.Choice, [1f, 2f], [1f]));
        Assert.Throws<ArgumentOutOfRangeException>(() => CalibrationSample.FromLabel(QuestionType.Choice, [1f, 2f], 2));
        Assert.Equal("choice:3-5", CalibrationSample.FromLabel(QuestionType.Choice, [1f, 2f, 3f], 0).Bucket);
    }
}
