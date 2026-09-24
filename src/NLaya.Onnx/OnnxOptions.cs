using Microsoft.ML.OnnxRuntime;

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
