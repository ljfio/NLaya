using Microsoft.Extensions.Logging;

namespace NLaya.TorchSharp;

/// <summary>The TorchSharp backend's log messages, source-generated.</summary>
internal static partial class TorchSharpLog
{
    [LoggerMessage(101, LogLevel.Warning, "laya: could not place the model on {Device}, so it is running on CPU. Reason: {Reason}")]
    public static partial void DeviceFallback(this ILogger logger, Exception exception, string device, string reason);

    [LoggerMessage(102, LogLevel.Warning, "laya: {Device} requested but not available. Falling back to CPU.")]
    public static partial void DeviceUnavailable(this ILogger logger, string device);
}
