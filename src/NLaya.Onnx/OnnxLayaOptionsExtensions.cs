namespace NLaya.Onnx;

/// <summary>Adds the ONNX Runtime backend to <see cref="LayaOptions"/>.</summary>
public static class OnnxLayaOptionsExtensions
{
    /// <summary>
    /// Run with ONNX Runtime from an <c>export_onnx.py</c> output directory. Load the agent from
    /// that same directory (it holds tokenizer.json and rl_agent_config.json), or from the Hub id
    /// the export was made from.
    /// </summary>
    public static LayaOptions UseOnnx(this LayaOptions options, string modelDir, bool useCuda = false) =>
        options.UseOnnx(modelDir, o => o.UseCuda = useCuda);

    /// <summary>
    /// Run with ONNX Runtime from <paramref name="modelDir"/>, with the rest of <see cref="OnnxOptions"/>
    /// (CUDA, session options, telemetry) set by <paramref name="configure"/>.
    /// </summary>
    public static LayaOptions UseOnnx(this LayaOptions options, string modelDir, Action<OnnxOptions> configure)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(configure);
        var o = new OnnxOptions { ModelDir = modelDir, Logger = options.Logger };
        configure(o);
        options.Backend = new OnnxBackendFactory(o);
        return options;
    }
}
