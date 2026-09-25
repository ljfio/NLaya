namespace NLaya.TorchSharp;

/// <summary>Adds the TorchSharp backend to <see cref="LayaOptions"/>.</summary>
public static class TorchSharpLayaOptionsExtensions
{
    /// <summary>
    /// Run with TorchSharp. Add a libtorch runtime package to your app: <c>TorchSharp-cpu</c>,
    /// <c>TorchSharp-cuda-linux</c> or <c>TorchSharp-cuda-windows</c>.
    /// </summary>
    public static LayaOptions UseTorchSharp(this LayaOptions options, Action<TorchSharpOptions>? configure = null)
    {
        var o = new TorchSharpOptions { Logger = options.Logger };
        configure?.Invoke(o);
        options.Backend = new TorchSharpBackendFactory(o);
        return options;
    }

    /// <summary>Run with TorchSharp on <paramref name="device"/> ("cpu", "cuda", "mps", "auto").</summary>
    public static LayaOptions UseTorchSharp(this LayaOptions options, string device) =>
        options.UseTorchSharp(o => o.Device = device);
}
