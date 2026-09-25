namespace NLaya.Tests.ML;

/// <summary>
/// Answers from the state's text alone: "urgent" in it makes the choice <c>support</c>, the noul
/// true and the score 2. Records each batch's states.
/// </summary>
internal sealed class TextPredictor : ILayaPredictor
{
    public List<List<string>> Batches { get; } = [];

    public LayaResult Predict(LayaState state, Questions questions, PredictOptions? options = null)
    {
        var urgent = state.Serialize().Contains("urgent", StringComparison.Ordinal);
        var p = urgent ? 0.8 : 0.3;
        var answers = new OrderedDictionary<string, Answer>
        {
            ["team"] = new ChoiceAnswer
            {
                Choice = urgent ? "support" : "billing",
                Probabilities = new() { ["billing"] = 1 - p, ["support"] = p },
                Confidence = 0.5,
                AnswerConfidence = Math.Max(p, 1 - p),
            },
            ["urgency"] = new ScoreAnswer
            {
                Score = urgent ? 2 : 0,
                Legend = [],
                Probabilities = urgent ? [0.1, 0.1, 0.8] : [0.8, 0.1, 0.1],
                Confidence = 0.5,
                AnswerConfidence = 0.8,
            },
            ["refund"] = new NoulAnswer { Probability = p, Confidence = Math.Max(p, 1 - p), AnswerConfidence = Math.Max(p, 1 - p) },
        };
        return new LayaResult(LayaResult.PythonModelName, answers, new Usage(1));
    }

    public Task<LayaResult> PredictAsync(LayaState state, Questions questions, PredictOptions? options = null, CancellationToken ct = default) =>
        Task.FromResult(Predict(state, questions, options));

    public IReadOnlyList<LayaResult> PredictBatch(IEnumerable<LayaState> states, Questions questions, BatchOptions? options = null)
    {
        var list = states.ToList();
        Batches.Add(list.Select(s => s.Serialize()).ToList());
        return list.Select(s => Predict(s, questions, options)).ToList();
    }
}
