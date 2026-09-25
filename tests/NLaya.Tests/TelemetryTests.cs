using System.Diagnostics;
using System.Diagnostics.Metrics;

using NLaya.Hub;
using NLaya.Tests.ExtensionsAI;

namespace NLaya.Tests;

/// <summary><see cref="LayaTelemetry"/>: spans and metrics reach standard .NET listeners (what OpenTelemetry subscribes with).</summary>
public class TelemetryTests
{
    [Fact]
    public void Load_and_predict_emit_spans_and_metrics()
    {
        if (HfCache.Snapshot(Laya.MultilingualModel) is null)
            Assert.Skip($"{Laya.MultilingualModel} is not cached; run: {HfCache.DownloadCommand(Laya.MultilingualModel)}");

        // Listeners are process-wide and other tests run in parallel: keep only spans under this test's
        // own root span, and only measurements for this test's model.
        using var testSource = new ActivitySource(nameof(TelemetryTests));
        var spans = new List<Activity>();
        using var activities = new ActivityListener
        {
            ShouldListenTo = s => s.Name is LayaTelemetry.Name or nameof(TelemetryTests),
            Sample = (ref ActivityCreationOptions<ActivityContext> _) => ActivitySamplingResult.AllDataAndRecorded,
            ActivityStopped = a =>
            {
                lock (spans) spans.Add(a);
            },
        };
        ActivitySource.AddActivityListener(activities);
        using var root = testSource.StartActivity("test")!;

        var measurements = new List<(string Name, double Value)>();
        using var meters = new MeterListener
        {
            InstrumentPublished = (i, l) =>
            {
                if (i.Meter.Name == LayaTelemetry.Name) l.EnableMeasurementEvents(i);
            },
        };
        void Record(Instrument i, double value, ReadOnlySpan<KeyValuePair<string, object?>> tags)
        {
            foreach (var t in tags)
                if (t.Key == "laya.model" && t.Value as string == Laya.MultilingualModel)
                    lock (measurements) measurements.Add((i.Name, value));
        }
        meters.SetMeasurementEventCallback<double>((i, v, tags, _) => Record(i, v, tags));
        meters.SetMeasurementEventCallback<long>((i, v, tags, _) => Record(i, v, tags));
        meters.Start();

        using var agent = Laya.Load(Laya.MultilingualModel, o => o.Backend = new FakeBackendFactory());
        var results = agent.PredictBatch(["one state", "and another"], Presets.Triage());

        lock (spans)
        {
            spans.RemoveAll(s => s.TraceId != root.TraceId);
            Assert.Contains(spans, s => s.OperationName == "laya.load");
            var predict = Assert.Single(spans, s => s.OperationName == "laya.predict");
            Assert.Equal(2, predict.GetTagItem("laya.states"));
            Assert.Equal((long)results.Sum(r => r.Usage.InputTokens), predict.GetTagItem("laya.input_tokens"));
        }
        lock (measurements)
        {
            Assert.Contains(measurements, m => m.Name == "laya.load.duration");
            Assert.Contains(measurements, m => m.Name == "laya.predict.duration");
            Assert.Contains(("laya.predict.states", 2.0), measurements);
            Assert.Contains(measurements, m => m.Name == "laya.predict.tokens" && m.Value > 0);
        }
    }
}
