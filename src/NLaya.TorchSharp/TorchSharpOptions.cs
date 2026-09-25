using Microsoft.Extensions.Logging;

using static TorchSharp.torch;

namespace NLaya.TorchSharp;

/// <summary>Settings for the TorchSharp backend.</summary>
public sealed class TorchSharpOptions
{
    /// <summary>
    /// "auto" (CUDA if available, else CPU), "cpu", "cuda", "cuda:1" or "mps". A device that is
    /// not available falls back to CPU with a warning, like the Python library.
    /// </summary>
    public string Device { get; set; } = "auto";

    /// <summary>Weight/compute precision. Null is float32; float16/bfloat16 are faster on GPUs.</summary>
    public ScalarType? DType { get; set; }

    /// <summary>
    /// CPU intra-op threads; null leaves libtorch's default. libtorch has one thread pool per process,
    /// so this applies to every agent, and the last backend built with it set wins. Prefer setting it
    /// once at startup (or <c>torch.set_num_threads</c>) when several agents share a process.
    /// </summary>
    public int? NumThreads { get; set; }

    /// <summary>Where device fallbacks are logged. <c>UseTorchSharp</c> copies <see cref="LayaOptions.Logger"/>.</summary>
    public ILogger? Logger { get; set; }
}
