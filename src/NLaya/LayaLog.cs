using Microsoft.Extensions.Logging;

namespace NLaya;

/// <summary>NLaya's log messages, source-generated so a disabled level costs no formatting or boxing.</summary>
internal static partial class LayaLog
{
    [LoggerMessage(1, LogLevel.Warning,
        "laya: this checkpoint ships invalid temperatures or values outside [{Min}, {Max}]; using {Values}. " +
        "Treat confidence from the affected entries as uncalibrated.")]
    public static partial void InvalidTemperatures(this ILogger logger, double min, double max, string values);

    [LoggerMessage(2, LogLevel.Warning, "laya: hook {Hook}.{Event} failed: {Message}")]
    public static partial void HookFailed(this ILogger logger, Exception exception, string hook, string @event, string message);

    [LoggerMessage(3, LogLevel.Warning, "laya: an error hook failed while handling {Error}")]
    public static partial void ErrorHookFailed(this ILogger logger, Exception exception, string error);

    [LoggerMessage(4, LogLevel.Warning, "laya: an end hook failed while handling {Error}")]
    public static partial void EndHookFailed(this ILogger logger, Exception exception, string error);
}
