using Microsoft.Extensions.Logging;

namespace NLaya.Onnx;

/// <summary>The ONNX backend's log messages, source-generated.</summary>
internal static partial class OnnxLog
{
    [LoggerMessage(201, LogLevel.Warning, "laya: the CUDA execution provider could not be added, so ONNX Runtime is running on CPU. Reason: {Reason}")]
    public static partial void CudaUnavailable(this ILogger logger, Exception exception, string reason);
}
