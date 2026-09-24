using Microsoft.ML.OnnxRuntime;
using NLaya.Backends;

namespace NLaya.Onnx;

/// <summary>Settings for the ONNX Runtime backend.</summary>
public sealed class OnnxOptions
{
    /// <summary>Directory holding <c>encoder.onnx</c> and <c>head.onnx</c> (from laya's <c>export_onnx.py</c>).</summary>
    public required string ModelDir { get; init; }

    /// <summary>Use the CUDA execution provider (needs Microsoft.ML.OnnxRuntime.Gpu); falls back to CPU.</summary>
    public bool UseCuda { get; set; }

    /// <summary>Tweak the session options (threads, other execution providers).</summary>
    public Action<SessionOptions>? Configure { get; set; }
}

/// <summary>Runs the split ONNX export (encoder.onnx then head.onnx) with ONNX Runtime.</summary>
public sealed class OnnxBackend : ILayaBackend
{
    private readonly InferenceSession _encoder;
    private readonly InferenceSession _head;
    private readonly bool _qtypeIs2D;

    public string Name { get; }

    public OnnxBackend(OnnxOptions options)
    {
        var enc = Path.Combine(options.ModelDir, "encoder.onnx");
        var head = Path.Combine(options.ModelDir, "head.onnx");
        if (!File.Exists(enc) || !File.Exists(head))
            throw new FileNotFoundException(
                $"ONNX model not found in '{options.ModelDir}'. Export it once with laya's laya-ts/scripts/export_onnx.py.");
        var cuda = false;
        SessionOptions Make()
        {
            var so = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
            if (options.UseCuda)
            {
                try { so.AppendExecutionProvider_CUDA(); cuda = true; }
                catch (Exception) { cuda = false; } // provider missing: keep CPU
            }
            options.Configure?.Invoke(so);
            return so;
        }
        using var encOptions = Make();
        using var headOptions = Make();
        _encoder = new InferenceSession(enc, encOptions);
        _head = new InferenceSession(head, headOptions);
        _qtypeIs2D = _head.InputMetadata["qtype"].Dimensions.Length == 2;
        Name = cuda ? "onnx:cuda" : "onnx:cpu";
    }

    public BackendOutput Run(EncodedBatch b)
    {
        long[] seq = [b.Rows, b.SeqLen], markers = [b.Rows, b.MaxMarkers];
        using var ids = OrtValue.CreateTensorValueFromMemory(b.InputIds, seq);
        using var att = OrtValue.CreateTensorValueFromMemory(b.AttentionMask, seq);
        using var encOut = _encoder.Run(new RunOptions(), ["input_ids", "attention_mask"], [ids, att], ["last_hidden_state"]);

        using var pos = OrtValue.CreateTensorValueFromMemory(b.MarkerPos, markers);
        using var mask = OrtValue.CreateTensorValueFromMemory(b.MarkerMask, markers);
        using var qt = OrtValue.CreateTensorValueFromMemory(b.QType, _qtypeIs2D ? [b.Rows, 1] : [b.Rows]);
        using var headOut = _head.Run(new RunOptions(),
            ["hidden_states", "marker_pos", "marker_mask", "qtype", "attention_mask"], [encOut[0], pos, mask, qt, att],
            ["logits", "act_logits"]);

        var act = headOut[1];
        return new BackendOutput(headOut[0].GetTensorDataAsSpan<float>().ToArray(), act.GetTensorDataAsSpan<float>().ToArray(),
            b.Rows, b.MaxMarkers, (int)act.GetTensorTypeAndShape().Shape[1]);
    }

    public void Dispose()
    {
        _encoder.Dispose();
        _head.Dispose();
    }
}

public sealed class OnnxBackendFactory(OnnxOptions options) : ILayaBackendFactory
{
    public IReadOnlyList<string> RequiredFiles { get; } = [];

    public ILayaBackend Create(LayaCheckpoint checkpoint) => new OnnxBackend(options);
}

public static class OnnxLayaOptionsExtensions
{
    /// <summary>
    /// Run with ONNX Runtime from an <c>export_onnx.py</c> output directory. Load the agent from
    /// that same directory (it holds tokenizer.json and rl_agent_config.json), or from the Hub id
    /// the export was made from.
    /// </summary>
    public static LayaOptions UseOnnx(this LayaOptions options, string modelDir, bool useCuda = false)
    {
        options.Backend = new OnnxBackendFactory(new OnnxOptions { ModelDir = modelDir, UseCuda = useCuda });
        return options;
    }
}
