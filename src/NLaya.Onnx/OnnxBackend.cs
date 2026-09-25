using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.ML.OnnxRuntime;

using NLaya.Backends;

namespace NLaya.Onnx;

/// <summary>Runs the split ONNX export (encoder.onnx then head.onnx) with ONNX Runtime.</summary>
public sealed class OnnxBackend : ILayaBackend
{
    private readonly InferenceSession _encoder;
    private readonly InferenceSession _head;
    private readonly bool _qtypeIs2D;
    // CUDA device memory for the encoder output, or null to let ONNX Runtime return it on the host.
    private readonly OrtMemoryInfo? _hiddenMemory;

    /// <inheritdoc/>
    public string Name { get; }

    /// <summary>Open <c>encoder.onnx</c> and <c>head.onnx</c> from <see cref="OnnxOptions.ModelDir"/>.</summary>
    public OnnxBackend(OnnxOptions options)
    {
        var enc = Path.Combine(options.ModelDir, "encoder.onnx");
        var head = Path.Combine(options.ModelDir, "head.onnx");
        if (!File.Exists(enc) || !File.Exists(head))
            throw new FileNotFoundException(
                $"ONNX model not found in '{options.ModelDir}'. Export it once with laya's laya-ts/scripts/export_onnx.py.");
        // Before any session exists, so no telemetry events are queued for the upload at exit.
        if (options.EnableTelemetry) OrtEnv.Instance().EnableTelemetryEvents();
        else OrtEnv.Instance().DisableTelemetryEvents();

        var log = options.Logger ?? NullLogger.Instance;
        var cuda = false;
        var warned = false;
        SessionOptions Make()
        {
            var so = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
            if (options.UseCuda)
            {
                // A missing provider (no Microsoft.ML.OnnxRuntime.Gpu, no CUDA) keeps CPU, like TorchSharp's fallback.
                try { so.AppendExecutionProvider_CUDA(options.CudaDeviceId); cuda = true; }
                catch (Exception e)
                {
                    cuda = false;
                    if (!warned) log.CudaUnavailable(e, e.Message);
                    warned = true;
                }
            }
            options.Configure?.Invoke(so);
            return so;
        }
        using var encOptions = Make();
        using var headOptions = Make();
        _encoder = new InferenceSession(enc, encOptions);
        _head = new InferenceSession(head, headOptions);
        _qtypeIs2D = _head.InputMetadata["qtype"].Dimensions.Length == 2;
        if (cuda && options.KeepHiddenStatesOnDevice)
            _hiddenMemory = new OrtMemoryInfo(OrtMemoryInfo.allocatorCUDA, OrtAllocatorType.DeviceAllocator, options.CudaDeviceId, OrtMemType.Default);
        Name = cuda ? "onnx:cuda" : "onnx:cpu";
    }

    /// <inheritdoc/>
    public BackendOutput Run(EncodedBatch batch)
    {
        using var run = new RunOptions();
        long[] seq = [batch.Rows, batch.SeqLen], markers = [batch.Rows, batch.MaxMarkers];
        using var ids = OrtValue.CreateTensorValueFromMemory(batch.InputIds, seq);
        using var att = OrtValue.CreateTensorValueFromMemory(batch.AttentionMask, seq);
        using var encOut = RunEncoder(run, ids, att);

        using var pos = OrtValue.CreateTensorValueFromMemory(batch.MarkerPos, markers);
        using var mask = OrtValue.CreateTensorValueFromMemory(batch.MarkerMask, markers);
        using var qt = OrtValue.CreateTensorValueFromMemory(batch.QType, _qtypeIs2D ? [batch.Rows, 1] : [batch.Rows]);
        using var headOut = _head.Run(run,
            ["hidden_states", "marker_pos", "marker_mask", "qtype", "attention_mask"], [encOut[0], pos, mask, qt, att],
            ["logits", "act_logits"]);

        var act = headOut[1];
        return new BackendOutput(headOut[0].GetTensorDataAsSpan<float>().ToArray(), act.GetTensorDataAsSpan<float>().ToArray(),
            batch.Rows, batch.MaxMarkers, (int)act.GetTensorTypeAndShape().Shape[1]);
    }

    /// <summary>The encoder's <c>last_hidden_state</c>; bound to device memory on CUDA so the head reads it in place.</summary>
    private IDisposableReadOnlyCollection<OrtValue> RunEncoder(RunOptions run, OrtValue ids, OrtValue att)
    {
        if (_hiddenMemory is null) return _encoder.Run(run, ["input_ids", "attention_mask"], [ids, att], ["last_hidden_state"]);
        using var binding = _encoder.CreateIoBinding();
        binding.BindInput("input_ids", ids);
        binding.BindInput("attention_mask", att);
        binding.BindOutputToDevice("last_hidden_state", _hiddenMemory);
        _encoder.RunWithBinding(run, binding);
        binding.SynchronizeBoundOutputs();
        return binding.GetOutputValues();
    }

    /// <summary>Release both ONNX Runtime sessions.</summary>
    public void Dispose()
    {
        _hiddenMemory?.Dispose();
        _encoder.Dispose();
        _head.Dispose();
    }
}
