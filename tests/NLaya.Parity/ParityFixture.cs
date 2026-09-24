using System.Text.Json.Nodes;
using NLaya.Backends;
using NLaya.Onnx;
using NLaya.TorchSharp;

namespace NLaya.Parity;

/// <summary>
/// Loads <c>convaiinnovations/laya-multilingual</c> once per test run. Model tests only run with
/// <c>NLAYA_PARITY=1</c>, because they download ~680 MB the first time.
/// </summary>
public sealed class ParityFixture : IDisposable
{
    public static bool Enabled => Environment.GetEnvironmentVariable("NLAYA_PARITY") is "1" or "true";
    public static string? OnnxDir => Environment.GetEnvironmentVariable("NLAYA_ONNX_DIR") is { Length: > 0 } d ? d : null;

    private readonly Lazy<LayaAgent> _torch = new(() =>
        Laya.Load(Laya.MultilingualModel, o => o.UseTorchSharp("cpu")));

    private readonly Lazy<LayaAgent?> _onnx = new(() => OnnxDir is { } dir
        ? Laya.Load(dir, o => o.UseOnnx(dir))
        : null);

    public LayaAgent Agent(string backend)
    {
        if (!Enabled) Assert.Skip("set NLAYA_PARITY=1 to run model parity tests");
        if (backend == "torchsharp") return _torch.Value;
        return _onnx.Value ?? throw SkipOnnx();
    }

    private static Exception SkipOnnx()
    {
        Assert.Skip("set NLAYA_ONNX_DIR to an export_onnx.py output directory to run ONNX parity tests");
        return new InvalidOperationException();
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
        if (_torch.IsValueCreated) _torch.Value.Dispose();
        if (_onnx.IsValueCreated) _onnx.Value?.Dispose();
    }
}
