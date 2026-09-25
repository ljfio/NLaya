using System.Text.Json.Nodes;

using NLaya.TorchSharp;

namespace NLaya.Parity;

[Collection(ParityCollection.Name)]
[TestCaseOrderer(typeof(ByModelOrderer))]
public class ModelParityTests(ParityFixture fx)
{
    // Reduced precision (NLAYA_DTYPE) drifts from float32 Python: labels must still agree (JsonAssert
    // compares strings exactly), probabilities within 0.1, and single hidden-state elements aren't compared.
    private static double LogitTol => ParityFixture.ReducedPrecision ? 0.5 : 2e-3;
    private static double ProbTol => ParityFixture.ReducedPrecision ? 0.1 : 1e-3;
    private static double HiddenMeanTol => ParityFixture.ReducedPrecision ? 0.02 : 1e-3;

    public static TheoryData<string, string, int> Cases(string section)
    {
        var data = new TheoryData<string, string, int>();
        foreach (var model in ParityFixture.Models)
        {
            var n = TestFiles.Fixture($"model_{model}.json")[section]!.AsArray().Count;
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
        var b = TestFiles.Fixture($"model_{model}.json")["batches"]![i]!;
        var agent = fx.Agent(backend, model);
        var batch = ParityFixture.Batch(b);
        var hidden = ((TorchSharpBackend)agent.Backend).EncodeHidden(batch);
        var d = agent.EncoderConfig!.HiddenSize;
        var head = b["hidden_head"]!.AsArray();
        for (var r = 0; r < head.Count && !ParityFixture.ReducedPrecision; r++)
            for (var t = 0; t < head[r]!.AsArray().Count; t++)
                for (var c = 0; c < head[r]![t]!.AsArray().Count; c++)
                    Assert.Equal(head[r]![t]![c]!.GetValue<double>(), hidden[(r * batch.SeqLen + t) * d + c], 2e-3);
        // Mean |h| over real tokens only would differ from Python's (which includes padding), so
        // compare it over the whole padded tensor, as Python computed it.
        Assert.Equal(b["hidden_mean_abs"]!.GetValue<double>(), hidden.Average(Math.Abs), HiddenMeanTol);
    }

    [Theory]
    [MemberData(nameof(Batches))]
    public void Logits_and_act_match(string backend, string model, int i)
    {
        var b = TestFiles.Fixture($"model_{model}.json")["batches"]![i]!;
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
        var c = TestFiles.Fixture($"model_{model}.json")["predicts"]![i]!;
        var agent = fx.Agent(backend, model);
        var result = agent.Predict(LayaState.FromJson(c["state"]?.DeepClone()), Questions.FromJson(c["questions"]));
        AssertResult(c["result"]!, result.ToJson());
    }

    [Theory]
    [MemberData(nameof(BackendModels))]
    public async Task PredictBatch_matches_python(string backend, string model)
    {
        var c = TestFiles.Fixture($"model_{model}.json")["predict_batch"]!;
        var agent = fx.Agent(backend, model);
        var states = c["states"]!.AsArray().Select(s => LayaState.FromJson(s?.DeepClone())).ToList();
        var results = agent.PredictBatch(states, Questions.FromJson(c["questions"]));
        var expected = c["results"]!.AsArray();
        Assert.Equal(expected.Count, results.Count);
        for (var j = 0; j < results.Count; j++) AssertResult(expected[j]!, results[j].ToJson());

        var sorted = agent.PredictBatch(states, Questions.FromJson(c["questions"]), new BatchOptions { BatchSize = 2, SortByLength = true });
        for (var j = 0; j < results.Count; j++) AssertResult(expected[j]!, sorted[j].ToJson());

        var ct = TestContext.Current.CancellationToken;
        var streamed = await agent.PredictStreamAsync(states, Questions.FromJson(c["questions"]), new BatchOptions { BatchSize = 2 }, ct).ToListAsync(ct);
        Assert.Equal(expected.Count, streamed.Count);
        for (var j = 0; j < streamed.Count; j++) AssertResult(expected[j]!, streamed[j].ToJson());
    }

    /// <summary>Same keys and strings; token counts exact, other numbers within <see cref="ProbTol"/>.</summary>
    internal static void AssertResult(JsonNode expected, JsonNode actual) =>
        JsonAssert.Equivalent(expected, actual, path => path.EndsWith("input_tokens", StringComparison.Ordinal) ? 0 : ProbTol);

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
