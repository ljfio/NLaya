using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

using NLaya.Backends;

using TorchSharp;

using static TorchSharp.torch;

namespace NLaya.TorchSharp;

public sealed class TorchSharpBackend : ILayaBackend
{
    private readonly LayaNetwork _net;
    private readonly ILogger _log;

    public string Name { get; }

    public TorchSharpBackend(LayaCheckpoint checkpoint, TorchSharpOptions options)
    {
        _log = options.Logger ?? NullLogger.Instance;
        var encCfg = checkpoint.EncoderConfig ?? throw new FileNotFoundException(
            $"'{checkpoint.ModelId}' has no encoder/config.json; the TorchSharp backend needs it to build the encoder.");
        if (!File.Exists(checkpoint.WeightsPath))
            throw new FileNotFoundException($"Incompatible model: 'model.safetensors' not found in '{checkpoint.ModelId}'.", checkpoint.WeightsPath);
        if (options.NumThreads is { } n) set_num_threads(n);

        var device = ResolveDevice(options.Device);
        var dtype = options.DType ?? ScalarType.Float32;
        using (no_grad())
        {
            Dictionary<string, Tensor> weights;
            try
            {
                weights = SafeTensors.Load(checkpoint.WeightsPath, dtype, device);
            }
            catch (Exception e) when (device.type != DeviceType.CPU && e is not IOException and not InvalidDataException)
            {
                _log.DeviceFallback(e, device.ToString(), e.Message);
                device = CPU;
                dtype = ScalarType.Float32;
                weights = SafeTensors.Load(checkpoint.WeightsPath, dtype, device);
            }
            weights.Remove("temperature", out var temp);
            temp?.Dispose();
            _net = new LayaNetwork(weights, encCfg, checkpoint.Config.HeadLayers, dtype, device);
        }
        Name = $"torchsharp:{device}";
    }

    private Device ResolveDevice(string spec)
    {
        spec = spec.Trim().ToLowerInvariant();
        if (spec == "auto") return cuda.is_available() ? CUDA : CPU;
        if (spec.StartsWith("cuda", StringComparison.Ordinal) && !cuda.is_available())
        {
            _log.DeviceUnavailable("CUDA");
            return CPU;
        }
        if (spec == "mps" && !mps_is_available())
        {
            _log.DeviceUnavailable("MPS");
            return CPU;
        }
        return torch.device(spec);
    }

    public BackendOutput Run(EncodedBatch batch)
    {
        using var mode = inference_mode();
        using var scope = NewDisposeScope();
        var dev = _net.Device;
        var (rows, len, k) = ((long)batch.Rows, (long)batch.SeqLen, (long)batch.MaxMarkers);
        var ids = tensor(batch.InputIds, [rows, len], device: dev);
        var att = tensor(batch.AttentionMask, [rows, len], device: dev);
        var mpos = tensor(batch.MarkerPos, [rows, k], device: dev);
        var mmask = tensor(batch.MarkerMask, [rows, k], device: dev);
        var qt = tensor(batch.QType, [rows], device: dev);
        var (logits, act) = _net.Forward(ids, att, mpos, mmask, qt, IsPadded(batch));
        var l = logits.cpu().data<float>().ToArray();
        var a = act.cpu().data<float>().ToArray();
        return new BackendOutput(l, a, batch.Rows, batch.MaxMarkers, (int)act.shape[1]);
    }

    /// <summary>Encoder hidden states [rows, len, hidden] for a batch (diagnostics and parity tests).</summary>
    public float[] EncodeHidden(EncodedBatch batch)
    {
        using var mode = inference_mode();
        using var scope = NewDisposeScope();
        var dev = _net.Device;
        var ids = tensor(batch.InputIds, [batch.Rows, batch.SeqLen], device: dev);
        var att = tensor(batch.AttentionMask, [batch.Rows, batch.SeqLen], device: dev);
        return _net.Encode(ids, att, IsPadded(batch)).to(ScalarType.Float32).cpu().data<float>().ToArray();
    }

    /// <summary>True when some row is shorter than the batch; without padding the network skips its masks.</summary>
    private static bool IsPadded(EncodedBatch batch) => batch.Lengths.AsSpan().ContainsAnyExcept(batch.SeqLen);

    public void Dispose() => _net.Dispose();
}
