namespace NLaya.Onnx;

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
