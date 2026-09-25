using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;

using NLaya;
using NLaya.Calibration;
using NLaya.Eval;
using NLaya.Onnx;
using NLaya.TorchSharp;

EvalArgs o;
try
{
    o = EvalArgs.Parse(args);
}
catch (ArgumentException e)
{
    Console.Error.WriteLine(e.Message);
    Console.Error.WriteLine(EvalArgs.Usage);
    return 2;
}

var load = Stopwatch.StartNew();
using var agent = Laya.Load(o.Model, l =>
{
    l.Subfolder = o.Subfolder;
    if (o.Onnx is not null) l.UseOnnx(o.Onnx, useCuda: o.Device != "cpu");
    else l.UseTorchSharp(t => { t.Device = o.Device; t.DType = o.DType; });
});
agent.Warmup();
Console.WriteLine($"{agent} loaded in {load.Elapsed.TotalSeconds:F1}s");
Console.WriteLine($"shipped temperatures: per type {Text(agent.Temperatures.PerType)}, by options {Text(agent.Temperatures.ByOptions)}");
Console.WriteLine();
Console.WriteLine($"{"suite",-26} {"n",6} {"acc",6} {"F1",6} {"ECE",6} {"Brier",6} {"NLL",6}  {"refit ECE",-16} {"p50 ms",7} {"p95 ms",7}");

var report = new JsonObject
{
    ["model"] = agent.ModelId,
    ["subfolder"] = o.Subfolder,
    ["backend"] = agent.Backend.Name,
    ["dtype"] = o.DType.ToString(),
    ["temperatures"] = new JsonObject { ["per_type"] = Json(agent.Temperatures.PerType), ["by_options"] = Json(agent.Temperatures.ByOptions) },
    ["suites"] = new JsonObject(),
};
var everything = new List<ScoredQuestion>();

foreach (var path in o.Suites)
{
    var name = Path.GetFileNameWithoutExtension(path);
    var cases = EvalCase.Load(path, o.Limit);
    var (scored, dropped) = Scorer.Score(agent, cases, o.BatchSize);
    everything.AddRange(scored);
    var shipped = Scorer.Shipped(agent, scored);
    var suite = new JsonObject
    {
        ["cases"] = cases.Count,
        ["dropped"] = dropped,
        ["shipped"] = Json(shipped),
        ["by_type"] = new JsonObject(scored.GroupBy(s => s.Type).OrderBy(g => g.Key)
            .Select(g => KeyValuePair.Create(g.Key.ToString().ToLowerInvariant(), (JsonNode?)Json(Scorer.Shipped(agent, g))))),
    };
    if (scored.Any(s => s.Workflow is not null))
        suite["by_workflow"] = new JsonObject(scored.GroupBy(s => s.Workflow ?? "").OrderBy(g => g.Key, StringComparer.Ordinal)
            .Select(g => KeyValuePair.Create(g.Key, (JsonNode?)Json(Scorer.Shipped(agent, g)))));

    var refitText = "";
    if (o.Refit is not null && scored.Count >= 60)
    {
        // As the harness does: fit on the first half, measure on the second, so the gain is out of sample.
        var half = scored.Count / 2;
        var fit = Fit(scored[..half], o.Refit);
        var held = scored[half..];
        var before = Scorer.Shipped(agent, held);
        var after = CalibrationMetrics.Evaluate(held.Select(s => s.At(Refitted(fit, s))));
        suite["refit"] = new JsonObject
        {
            ["mode"] = o.Refit,
            ["held_out"] = held.Count,
            ["fitted"] = new JsonObject { ["per_type"] = Json(fit.PerType), ["by_options"] = Json(fit.ByOptions) },
            ["shipped"] = Json(before),
            ["refit"] = Json(after),
        };
        refitText = $"{before.Ece:F3} -> {after.Ece:F3}";
    }

    var (p50, p95) = (double.NaN, double.NaN);
    if (o.Latency > 0)
    {
        (p50, p95) = Latency(agent, cases, o.Latency);
        suite["latency_ms"] = new JsonObject { ["p50"] = p50, ["p95"] = p95, ["calls"] = Math.Min(o.Latency, cases.Count), ["questions_per_call"] = 1 };
    }

    report["suites"]![name] = suite;
    Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
        $"{name,-26} {shipped.Count,6} {shipped.Accuracy,6:F3} {shipped.MacroF1,6:F3} {shipped.Ece,6:F3} {shipped.Brier,6:F3} {shipped.Nll,6:F3}  {refitText,-16} {p50,7:F1} {p95,7:F1}"));
    if (suite["by_workflow"] is JsonObject wf)
        foreach (var (w, m) in wf) Console.WriteLine($"  {w,-24} {m!["count"],6} {m["accuracy"]!.GetValue<double>(),6:F3}");
}

if (o.SaveConfig is not null)
{
    var fit = Fit(everything, o.Refit ?? "per-type");
    var source = Path.Combine(agent.Directory, "rl_agent_config.json");
    fit.SaveConfig(source, o.SaveConfig);
    Console.WriteLine();
    Console.WriteLine($"fitted on {everything.Count} answers: per type {Text(fit.PerType)}, by options {Text(fit.ByOptions)}");
    var clamped = fit.PerType.Concat(fit.ByOptions.Values).Where(t => TemperatureTable.Clamp(t) != t).ToList();
    if (clamped.Count > 0)
        Console.WriteLine($"note: {Text(clamped)} lie outside [{TemperatureTable.Min}, {TemperatureTable.Max}]; Laya clamps them when it loads the config");
    Console.WriteLine($"wrote {o.SaveConfig}; copy the checkpoint directory and put it there as rl_agent_config.json");
}

if (o.Report is not null)
{
    File.WriteAllText(o.Report, report.ToJsonString(new JsonSerializerOptions
    {
        WriteIndented = true,
        Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
    }));
    Console.WriteLine($"wrote {o.Report}");
}
return 0;

// Per bucket, as the harness refits (grid, labels); per type, as the notebook fits (exact, teacher distributions).
TemperatureFit Fit(IReadOnlyList<ScoredQuestion> scored, string mode) => mode == "per-bucket"
    ? TemperatureFitter.FitPerBucket(scored.Select(s => s.Sample(soft: false)), TemperatureFitMethod.Grid, agent.Temperatures.PerType)
    : TemperatureFitter.FitPerType(scored.Select(s => s.Sample(soft: true)), TemperatureFitMethod.Optimize, agent.Temperatures.PerType);

// What an agent loading the fitted config would apply: bucket, else type (a per-type fit drops buckets), clamped at load.
double Refitted(TemperatureFit fit, ScoredQuestion s) => TemperatureTable.Clamp(
    fit.ByOptions.TryGetValue(s.Bucket, out var t) ? t
    : fit.ByOptions.Count == 0 ? fit.PerType[(int)s.Type]
    : agent.Temperatures.For(s.Type, s.Logits.Length));

// Laya's "1 question" latency: one state, its first question, one call at a time.
static (double P50, double P95) Latency(LayaAgent agent, List<EvalCase> cases, int calls)
{
    var times = new List<double>();
    foreach (var c in cases.Take(calls))
    {
        var (id, q) = c.Questions.First();
        var one = new Questions { [id] = q };
        var sw = Stopwatch.StartNew();
        agent.Predict(c.State, one);
        times.Add(sw.Elapsed.TotalMilliseconds);
    }
    times.Sort();
    double At(double p) => times[Math.Min(times.Count - 1, (int)(p * times.Count))];
    return (At(0.5), At(0.95));
}

static JsonNode Json<T>(T value) => JsonSerializer.SerializeToNode(value, EvalJson.Default.Options)!;

static string Text<T>(T value) => Json(value).ToJsonString(new JsonSerializerOptions { Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping });
