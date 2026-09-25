using Microsoft.ML.OnnxRuntime;

namespace NLaya.Onnx;

/// <summary>Settings for the ONNX Runtime backend.</summary>
public sealed class OnnxOptions
{
    /// <summary>Directory holding <c>encoder.onnx</c> and <c>head.onnx</c> (from laya's <c>export_onnx.py</c>).</summary>
    public required string ModelDir { get; init; }

    /// <summary>Use the CUDA execution provider (needs Microsoft.ML.OnnxRuntime.Gpu); falls back to CPU.</summary>
    public bool UseCuda { get; set; }

    /// <summary>
    /// Let ONNX Runtime send its own usage telemetry to Microsoft. Off by default: a library shouldn't
    /// phone home, and on macOS the upload thread can crash the process at exit (it locks a mutex that
    /// exit has already destroyed). The setting is process-wide, so the last backend created decides.
    /// </summary>
    public bool EnableTelemetry { get; set; }

    /// <summary>Tweak the session options (threads, other execution providers).</summary>
    public Action<SessionOptions>? Configure { get; set; }
}
