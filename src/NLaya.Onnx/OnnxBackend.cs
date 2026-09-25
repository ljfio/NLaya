using Microsoft.ML.OnnxRuntime;

using NLaya.Backends;

namespace NLaya.Onnx;

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

    public BackendOutput Run(EncodedBatch batch)
    {
        using var run = new RunOptions();
        long[] seq = [batch.Rows, batch.SeqLen], markers = [batch.Rows, batch.MaxMarkers];
        using var ids = OrtValue.CreateTensorValueFromMemory(batch.InputIds, seq);
        using var att = OrtValue.CreateTensorValueFromMemory(batch.AttentionMask, seq);
        using var encOut = _encoder.Run(run, ["input_ids", "attention_mask"], [ids, att], ["last_hidden_state"]);

        using var pos = OrtValue.CreateTensorValueFromMemory(batch.MarkerPos, markers);
        using var mask = OrtValue.CreateTensorValueFromMemory(batch.MarkerMask, markers);
        using var qt = OrtValue.CreateTensorValueFromMemory(batch.QType, _qtypeIs2D ? [batch.Rows, 1] : [batch.Rows]);
        using var headOut = _head.Run(run,
            ["hidden_states", "marker_pos", "marker_mask", "qtype", "attention_mask"], [encOut[0], pos, mask, qt, att],
            ["logits", "act_logits"]);

        var act = headOut[1];
        return new BackendOutput(headOut[0].GetTensorDataAsSpan<float>().ToArray(), act.GetTensorDataAsSpan<float>().ToArray(),
            batch.Rows, batch.MaxMarkers, (int)act.GetTensorTypeAndShape().Shape[1]);
    }

    public void Dispose()
    {
        _encoder.Dispose();
        _head.Dispose();
    }
}
