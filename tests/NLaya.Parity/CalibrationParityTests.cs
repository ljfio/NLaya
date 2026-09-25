using NLaya.Calibration;

namespace NLaya.Parity;

/// <summary><see cref="LayaAgent.PredictLogits"/> is what <c>Predict</c> scales: softmax(logits / T) gives its probabilities.</summary>
[Collection(ParityCollection.Name)]
[TestCaseOrderer(typeof(ByModelOrderer))]
public class CalibrationParityTests(ParityFixture fx)
{
    [Theory]
    [MemberData(nameof(ModelParityTests.BackendModels), MemberType = typeof(ModelParityTests))]
    public void PredictLogits_scaled_by_temperature_give_Predicts_probabilities(string backend, string model)
    {
        var c = TestFiles.Fixture($"model_{model}.json")["predict_batch"]!;
        var agent = fx.Agent(backend, model);
        var states = c["states"]!.AsArray().Select(s => LayaState.FromJson(s?.DeepClone())).ToList();
        var questions = Questions.FromJson(c["questions"]);

        var results = agent.PredictBatch(states, questions);
        var logits = agent.PredictLogits(states, questions, new BatchOptions { BatchSize = 2, SortByLength = true });

        for (var i = 0; i < states.Count; i++)
        {
            foreach (var (id, q) in questions)
            {
                var z = logits[i][id];
                Assert.Equal(q.OptionCount, z.Length);
                var p = CalibrationMetrics.Softmax(z, agent.Temperatures.For(q.Type, z.Length));
                double[] expected = results[i].Answers[id] switch
                {
                    ChoiceAnswer a => [.. a.Probabilities.Values],
                    ScoreAnswer a => [.. a.Probabilities],
                    NoulAnswer a => [1 - a.Probability, a.Probability],
                    _ => throw new InvalidOperationException(),
                };
                // Predict rounds to 4 places; reduced precision also drifts between batchings.
                var tol = ParityFixture.ReducedPrecision ? 0.1 : 1e-4;
                for (var k = 0; k < p.Length; k++) Assert.Equal(expected[k], p[k], tol);
            }
        }
    }
}
