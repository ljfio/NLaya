namespace NLaya.Tests.ExtensionsAI;

/// <summary>An <see cref="ILayaPredictor"/> that answers from a delegate and records what it was asked.</summary>
internal sealed class FakePredictor(Func<string, Questions, OrderedDictionary<string, Answer>> answer) : ILayaPredictor
{
    public List<(string Text, Questions Questions)> Calls { get; } = [];

    public LayaResult Predict(LayaState state, Questions questions, PredictOptions? options = null)
    {
        Calls.Add((state.Serialize(), questions));
        return new LayaResult(LayaResult.PythonModelName, answer(state.Serialize(), questions), new Usage(1));
    }

    public Task<LayaResult> PredictAsync(LayaState state, Questions questions, PredictOptions? options = null, CancellationToken ct = default) =>
        Task.FromResult(Predict(state, questions, options));

    public static NoulAnswer Noul(double p) => new() { Probability = p, Confidence = Math.Max(p, 1 - p), AnswerConfidence = Math.Max(p, 1 - p) };

    public static ScoreAnswer Score(double score) =>
        new() { Score = score, Legend = [], Probabilities = [1.0], Confidence = 0.5, AnswerConfidence = 0.5 };

    public static ChoiceAnswer Choice(string choice, double confidence = 0.9, double answerConfidence = 0.9) =>
        new() { Choice = choice, Probabilities = new() { [choice] = answerConfidence }, Confidence = confidence, AnswerConfidence = answerConfidence };
}
