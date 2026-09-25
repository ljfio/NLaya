using System.Diagnostics;
using System.Diagnostics.Metrics;

namespace NLaya;

/// <summary>
/// Tracing and metrics through the .NET diagnostics APIs, which OpenTelemetry collects: an
/// <see cref="ActivitySource"/> and a <see cref="Meter"/>, both named <see cref="Name"/>
/// (<c>.AddSource("NLaya")</c>, <c>.AddMeter("NLaya")</c>). Nothing is recorded until a listener subscribes.
/// </summary>
/// <remarks>
/// Spans: <c>laya.load</c> (a checkpoint load) and <c>laya.predict</c> (one agent call, over one or more
/// states). Metrics: <c>laya.load.duration</c> and <c>laya.predict.duration</c> (seconds),
/// <c>laya.predict.states</c>, <c>laya.predict.tokens</c> (input tokens), <c>laya.route.decisions</c> and
/// <c>laya.microbatch.size</c> (requests per <see cref="MicroBatchingPredictor"/> batch, untagged).
/// Tags: <c>laya.model</c> (the model id, or the Router's checkpoint name for routes) and, on failure,
/// <c>error.type</c>.
/// </remarks>
public static class LayaTelemetry
{
    public const string Name = "NLaya";

    internal const string ModelTag = "laya.model";

    internal static readonly ActivitySource Source = new(Name);
    private static readonly Meter Meter = new(Name);

    private static readonly Histogram<double> LoadDuration = Meter.CreateHistogram<double>(
        "laya.load.duration", "s", "Time to load a checkpoint and build its backend.");
    private static readonly Histogram<double> PredictDuration = Meter.CreateHistogram<double>(
        "laya.predict.duration", "s", "Time for one agent call, hooks included.");
    private static readonly Counter<long> PredictStates = Meter.CreateCounter<long>(
        "laya.predict.states", "{state}", "States answered.");
    private static readonly Counter<long> PredictTokens = Meter.CreateCounter<long>(
        "laya.predict.tokens", "{token}", "Input tokens read by the model.");
    private static readonly Histogram<int> MicroBatchSize = Meter.CreateHistogram<int>(
        "laya.microbatch.size", "{request}", "Requests answered together by a MicroBatchingPredictor batch.");
    private static readonly Counter<long> RouteDecisions = Meter.CreateCounter<long>(
        "laya.route.decisions", "{decision}", "Router decisions, by checkpoint.");

    /// <summary>Runs <paramref name="load"/> inside a <c>laya.load</c> span and records its duration.</summary>
    internal static T Load<T>(string model, Func<T> load)
    {
        using var activity = Source.StartActivity("laya.load");
        activity?.SetTag(ModelTag, model);
        var started = Stopwatch.GetTimestamp();
        string? error = null;
        try
        {
            return load();
        }
        catch (Exception ex)
        {
            error = Failed(activity, ex);
            throw;
        }
        finally
        {
            if (LoadDuration.Enabled) LoadDuration.Record(Stopwatch.GetElapsedTime(started).TotalSeconds, Tags(model, error));
        }
    }

    /// <summary>Runs <paramref name="predict"/> inside a <c>laya.predict</c> span and records its duration, states and tokens.</summary>
    internal static IList<LayaResult> Predict(string model, int states, Func<IList<LayaResult>> predict)
    {
        using var activity = Source.StartActivity("laya.predict");
        activity?.SetTag(ModelTag, model);
        activity?.SetTag("laya.states", states);
        var started = Stopwatch.GetTimestamp();
        string? error = null;
        try
        {
            var results = predict();
            if (activity is not null || PredictTokens.Enabled)
            {
                var tokens = results.Sum(r => (long)r.Usage.InputTokens);
                activity?.SetTag("laya.input_tokens", tokens);
                PredictTokens.Add(tokens, new KeyValuePair<string, object?>(ModelTag, model));
            }
            PredictStates.Add(results.Count, new KeyValuePair<string, object?>(ModelTag, model));
            return results;
        }
        catch (Exception ex)
        {
            error = Failed(activity, ex);
            throw;
        }
        finally
        {
            if (PredictDuration.Enabled) PredictDuration.Record(Stopwatch.GetElapsedTime(started).TotalSeconds, Tags(model, error));
        }
    }

    internal static void MicroBatch(int size)
    {
        if (MicroBatchSize.Enabled) MicroBatchSize.Record(size);
    }

    internal static void Routed(string checkpoint) =>
        RouteDecisions.Add(1, new KeyValuePair<string, object?>(ModelTag, checkpoint));

    private static string Failed(Activity? activity, Exception ex)
    {
        var type = ex.GetType().FullName ?? ex.GetType().Name;
        activity?.SetStatus(ActivityStatusCode.Error, ex.Message);
        activity?.AddException(ex);
        activity?.SetTag("error.type", type);
        return type;
    }

    private static TagList Tags(string model, string? error)
    {
        var tags = new TagList { { ModelTag, model } };
        if (error is not null) tags.Add("error.type", error);
        return tags;
    }
}
