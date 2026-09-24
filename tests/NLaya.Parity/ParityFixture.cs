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

    private readonly Dictionary<string, LayaAgent> _agents = new();

    public LayaAgent Agent(string backend, string model)
    {
        if (!Enabled) Assert.Skip("set NLAYA_PARITY=1 to run model parity tests");
        var onnxDir = OnnxRoot is null ? null : Path.Combine(OnnxRoot, model);
        if (backend == "onnx" && (onnxDir is null || !File.Exists(Path.Combine(onnxDir, "encoder.onnx"))))
            Assert.Skip($"set NLAYA_ONNX_ROOT to a directory with an export_onnx.py output in {model}/ to run ONNX parity tests");
        lock (_agents)
        {
            var key = backend + ":" + model;
            if (_agents.TryGetValue(key, out var agent)) return agent;
            var f = Fixture($"model_{model}.json");
            var repo = f["repo"]!.GetValue<string>();
            var sub = f["subfolder"]?.GetValue<string>();
            agent = backend == "onnx"
                ? Laya.Load(onnxDir!, o => o.UseOnnx(onnxDir!))
                : Laya.Load(repo, o => { o.Subfolder = sub; o.UseTorchSharp("cpu"); });
            return _agents[key] = agent;
        }
    }

    public static JsonNode Fixture(string name) =>
        JsonNode.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Fixtures", name)))!;

    public static EncodedBatch Batch(JsonNode b)
    {
        long[][] L(string k) => b[k]!.AsArray().Select(r => r!.AsArray().Select(x => x!.GetValue<long>()).ToArray()).ToArray();
        var mask = b["marker_mask"]!.AsArray().Select(r => r!.AsArray().Select(x => x!.GetValue<bool>()).ToArray()).ToArray();
        return EncodedBatch.FromArrays(L("input_ids"), L("attention_mask"), L("marker_pos"), mask,
            b["qtype"]!.AsArray().Select(x => x!.GetValue<long>()).ToArray());
    }

    public void Dispose()
    {
        foreach (var a in _agents.Values) a.Dispose();
    }
}
