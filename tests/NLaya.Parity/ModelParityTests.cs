using System.Text.Json.Nodes;

using NLaya.TorchSharp;

namespace NLaya.Parity;

public class ModelParityTests(ParityFixture fx) : IClassFixture<ParityFixture>
{
    private const double LogitTol = 2e-3;
    private const double ProbTol = 1e-3;

    public static TheoryData<string, string, int> Cases(string section)
    {
        var data = new TheoryData<string, string, int>();
        foreach (var model in ParityFixture.Models)
        {
            var n = ParityFixture.Fixture($"model_{model}.json")[section]!.AsArray().Count;
            foreach (var backend in new[] { "torchsharp", "onnx" })
                for (var i = 0; i < n; i++) data.Add(backend, model, i);
        }
        return data;
    }

    public static TheoryData<string, string, int> Batches() => Cases("batches");
    public static TheoryData<string, string, int> Predicts() => Cases("predicts");

    public static TheoryData<string, string> BackendModels()
    {
        var data = new TheoryData<string, string>();
        foreach (var model in ParityFixture.Models)
            foreach (var backend in new[] { "torchsharp", "onnx" }) data.Add(backend, model);
        return data;
    }

    [Theory]
    [MemberData(nameof(Batches))]
    public void Encoder_hidden_states_match(string backend, string model, int i)
    {
        if (backend != "torchsharp") Assert.Skip("hidden states are only exposed by the TorchSharp backend");
        var b = ParityFixture.Fixture($"model_{model}.json")["batches"]![i]!;
        var agent = fx.Agent(backend, model);
        var batch = ParityFixture.Batch(b);
        var hidden = ((TorchSharpBackend)agent.Backend).EncodeHidden(batch);
        var d = agent.EncoderConfig!.HiddenSize;
        var head = b["hidden_head"]!.AsArray();
        for (var r = 0; r < head.Count; r++)
            for (var t = 0; t < head[r]!.AsArray().Count; t++)
                for (var c = 0; c < head[r]![t]!.AsArray().Count; c++)
                    Assert.Equal(head[r]![t]![c]!.GetValue<double>(), hidden[(r * batch.SeqLen + t) * d + c], 2e-3);
        // Mean |h| over real tokens only would differ from Python's (which includes padding), so
        // compare it over the whole padded tensor, as Python computed it.
        Assert.Equal(b["hidden_mean_abs"]!.GetValue<double>(), hidden.Average(Math.Abs), 1e-3);
    }

    [Theory]
    [MemberData(nameof(Batches))]
    public void Logits_and_act_match(string backend, string model, int i)
    {
        var b = ParityFixture.Fixture($"model_{model}.json")["batches"]![i]!;
        var agent = fx.Agent(backend, model);
        var batch = ParityFixture.Batch(b);
        var output = agent.Backend.Run(batch);

        var logits = b["logits"]!.AsArray();
        for (var r = 0; r < batch.Rows; r++)
            for (var k = 0; k < batch.MaxMarkers; k++)
                Assert.Equal(logits[r]![k]!.GetValue<double>(), output.Logits[r * batch.MaxMarkers + k], LogitTol);

        // act logits are ~1e3 and only read through a softmax: compare probabilities (as export_onnx.py does).
        var act = b["act_logits"]!.AsArray();
        var expected = act.SelectMany(r => r!.AsArray().Select(x => (float)x!.GetValue<double>())).ToArray();
        var want = Softmax(expected, batch.Rows, output.ActOutputs);
        var got = Softmax(output.ActLogits, batch.Rows, output.ActOutputs);
        for (var j = 0; j < want.Length; j++) Assert.Equal(want[j], got[j], ProbTol);
    }

    [Theory]
    [MemberData(nameof(Predicts))]
    public void Predict_matches_python(string backend, string model, int i)
    {
        var c = ParityFixture.Fixture($"model_{model}.json")["predicts"]![i]!;
        var agent = fx.Agent(backend, model);
        var result = agent.Predict(LayaState.FromJson(c["state"]?.DeepClone()), Questions.FromJson(c["questions"]));
        AssertResult(c["result"]!, result.ToJson());
    }

    [Theory]
    [MemberData(nameof(BackendModels))]
    public void PredictBatch_matches_python(string backend, string model)
    {
        var c = ParityFixture.Fixture($"model_{model}.json")["predict_batch"]!;
        var agent = fx.Agent(backend, model);
        var states = c["states"]!.AsArray().Select(s => LayaState.FromJson(s?.DeepClone())).ToList();
        var results = agent.PredictBatch(states, Questions.FromJson(c["questions"]));
        var expected = c["results"]!.AsArray();
        Assert.Equal(expected.Count, results.Count);
        for (var j = 0; j < results.Count; j++) AssertResult(expected[j]!, results[j].ToJson());

        var sorted = agent.PredictBatch(states, Questions.FromJson(c["questions"]), new BatchOptions { BatchSize = 2, SortByLength = true });
        for (var j = 0; j < results.Count; j++) AssertResult(expected[j]!, sorted[j].ToJson());
    }

    /// <summary>Same keys and strings; numbers within <see cref="ProbTol"/>.</summary>
    private static void AssertResult(JsonNode expected, JsonNode? actual, string path = "$")
    {
        switch (expected)
        {
            case JsonObject eo:
            {
                var ao = Assert.IsType<JsonObject>(actual);
                Assert.Equal(eo.Select(kv => kv.Key), ao.Select(kv => kv.Key));
                foreach (var (k, v) in eo) AssertResult(v!, ao[k], $"{path}.{k}");
                break;
            }
            case JsonArray ea:
            {
                var aa = Assert.IsType<JsonArray>(actual);
                Assert.Equal(ea.Count, aa.Count);
                for (var i = 0; i < ea.Count; i++) AssertResult(ea[i]!, aa[i], $"{path}[{i}]");
                break;
            }
            case JsonValue ev when ev.GetValueKind() == System.Text.Json.JsonValueKind.Number:
            {
                var e = Number(ev);
                var a = Number(actual!);
                // input_tokens must match exactly; probabilities within tolerance.
                Assert.True(Math.Abs(e - a) <= (path.EndsWith("input_tokens") ? 0 : ProbTol), $"{path}: expected {e}, got {a}");
                break;
            }
            default:
                Assert.True(JsonNode.DeepEquals(expected, actual), $"{path}: expected {expected.ToJsonString()}, got {actual?.ToJsonString()}");
                break;
        }
    }

    private static double Number(JsonNode n) =>
        double.Parse(n.ToJsonString(), System.Globalization.CultureInfo.InvariantCulture);

    private static float[] Softmax(float[] x, int rows, int cols)
    {
        var o = new float[x.Length];
        for (var r = 0; r < rows; r++)
        {
            var max = x.Skip(r * cols).Take(cols).Max();
            var sum = 0f;
            for (var c = 0; c < cols; c++) sum += o[r * cols + c] = MathF.Exp(x[r * cols + c] - max);
            for (var c = 0; c < cols; c++) o[r * cols + c] /= sum;
        }
        return o;
    }
}
