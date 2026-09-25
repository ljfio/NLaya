using System.Text.Json.Nodes;

using NLaya.Backends;
using NLaya.Onnx;
using NLaya.TorchSharp;

namespace NLaya.Parity;

/// <summary>
/// Loads each checkpoint once per test run, from the Hugging Face cache. Model tests only run with
/// <c>NLAYA_PARITY=1</c>: they need <c>convaiinnovations/laya</c> (english, typed-decisions/) and
/// <c>convaiinnovations/laya-multilingual</c> fetched with <c>hf download</c>, ~2.5 GB.
/// </summary>
public sealed class ParityFixture : IDisposable
{
    public static bool Enabled => Environment.GetEnvironmentVariable("NLAYA_PARITY") is "1" or "true";
    /// <summary>A directory holding one export_onnx.py output per checkpoint: multilingual/, english/, typed_decisions/.</summary>
    public static string? OnnxRoot => Environment.GetEnvironmentVariable("NLAYA_ONNX_ROOT") is { Length: > 0 } d ? d : null;

    /// <summary>Fixture name -> (repo, subfolder), as make_fixtures.py generated them.</summary>
    public static readonly string[] Models = ["multilingual", "english", "typed_decisions"];

    // One checkpoint at a time: all of them at once (~20 GB peak) doesn't fit a CI runner.
    // ByModelOrderer keeps each class's cases grouped by checkpoint, so each loads about once.
    private readonly Lock _lock = new();
    private (string Key, LayaAgent Agent)? _current;

    public LayaAgent Agent(string backend, string model)
    {
        if (!Enabled) Assert.Skip("set NLAYA_PARITY=1 to run model parity tests");
        var onnxDir = OnnxRoot is null ? null : Path.Combine(OnnxRoot, model);
        if (backend == "onnx" && (onnxDir is null || !File.Exists(Path.Combine(onnxDir, "encoder.onnx"))))
            Assert.Skip($"set NLAYA_ONNX_ROOT to a directory with an export_onnx.py output in {model}/ to run ONNX parity tests");
        lock (_lock)
        {
            var key = backend + ":" + model;
            if (_current is { } c && c.Key == key) return c.Agent;
            _current?.Agent.Dispose();
            _current = null;
            GC.Collect();
            var f = TestFiles.Fixture($"model_{model}.json");
            var repo = f["repo"]!.GetValue<string>();
            var sub = f["subfolder"]?.GetValue<string>();
            var agent = backend == "onnx"
                ? Laya.Load(onnxDir!, o => o.UseOnnx(onnxDir!))
                : Laya.Load(repo, o => { o.Subfolder = sub; o.UseTorchSharp("cpu"); });
            _current = (key, agent);
            return agent;
        }
    }

    public static EncodedBatch Batch(JsonNode b)
    {
        long[][] L(string k) => b[k]!.AsArray().Select(r => r!.AsArray().Select(x => x!.GetValue<long>()).ToArray()).ToArray();
        var mask = b["marker_mask"]!.AsArray().Select(r => r!.AsArray().Select(x => x!.GetValue<bool>()).ToArray()).ToArray();
        return EncodedBatch.FromArrays(L("input_ids"), L("attention_mask"), L("marker_pos"), mask,
            b["qtype"]!.AsArray().Select(x => x!.GetValue<long>()).ToArray());
    }

    public void Dispose()
    {
        _current?.Agent.Dispose();
    }
}
