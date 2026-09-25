using BenchmarkDotNet.Attributes;

using NLaya.TorchSharp;

using static TorchSharp.torch;

namespace NLaya.Benchmarks;

/// <summary>
/// Latency and throughput of one agent, in the shapes laya reports: one question, ten questions
/// about one state, a 32-state batch, and 32 concurrent single requests through the micro-batcher.
/// Checkpoints are read from the Hugging Face cache.
/// </summary>
[MemoryDiagnoser(displayGenColumns: false)]
[SimpleJob(launchCount: 1, warmupCount: 3, iterationCount: 10)]
public class PredictBenchmarks : IDisposable
{
    private const string Ticket = "I was charged twice for invoice 4411 and nobody has answered my emails for a week. " +
        "Please refund the duplicate charge today or I will cancel the subscription.";

    private LayaAgent _agent = null!;
    private MicroBatchingPredictor _batcher = null!;
    private Questions _one = null!;
    private Questions _ten = null!;
    private LayaState[] _states = null!;

    /// <summary>"multilingual" (mmBERT-base) or "english" (ModernBERT-large).</summary>
    [Params("multilingual", "english")]
    public string Model { get; set; } = "multilingual";

    [GlobalSetup]
    public void Setup()
    {
        var device = Environment.GetEnvironmentVariable("NLAYA_DEVICE") is { Length: > 0 } d ? d : "cpu";
        var dtype = Environment.GetEnvironmentVariable("NLAYA_DTYPE") switch
        {
            "bfloat16" => ScalarType.BFloat16,
            "float16" => ScalarType.Float16,
            _ => ScalarType.Float32,
        };
        _agent = Laya.Load(Laya.DefaultModel, o =>
        {
            o.Subfolder = Model == "multilingual" ? "multilingual" : null;
            o.UseTorchSharp(t => { t.Device = device; t.DType = dtype; });
        });
        _agent.Warmup();
        _batcher = new MicroBatchingPredictor(_agent, new MicroBatchOptions { MaxBatchSize = 32 });
        _one = new Questions().Noul("refund_requested", "Does the sender ask for money back?");
        _ten = new Questions();
        for (var i = 0; i < 10; i++) _ten.Noul($"q{i}", $"Question {i}: does the sender mention topic {i}?");
        _states = Enumerable.Range(0, 32).Select(i => (LayaState)$"{Ticket} (ticket {i})").ToArray();
    }

    [GlobalCleanup]
    public void Dispose()
    {
        _batcher?.Dispose();
        _agent?.Dispose();
        GC.SuppressFinalize(this);
    }

    /// <summary>Laya's headline number: one state, one question.</summary>
    [Benchmark(Baseline = true)]
    public LayaResult OneQuestion() => _agent.Predict(Ticket, _one);

    /// <summary>Ten questions about one state share a forward pass (ten rows).</summary>
    [Benchmark]
    public LayaResult TenQuestions() => _agent.Predict(Ticket, _ten);

    /// <summary>32 states, one question, one pass.</summary>
    [Benchmark(OperationsPerInvoke = 32)]
    public IReadOnlyList<LayaResult> Batch32() => _agent.PredictBatch(_states, _one);

    /// <summary>32 concurrent callers with one state each, coalesced by the micro-batcher.</summary>
    [Benchmark(OperationsPerInvoke = 32)]
    public Task<LayaResult[]> MicroBatched32() => Task.WhenAll(_states.Select(s => _batcher.PredictAsync(s, _one)));
}
