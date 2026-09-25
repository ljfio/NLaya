using NLaya.Calibration;

namespace NLaya.Eval;

/// <summary>Runs a suite through <see cref="LayaAgent.PredictLogits"/>, cases with the same questions sharing passes.</summary>
internal static class Scorer
{
    public static (List<ScoredQuestion> Scored, int Dropped) Score(LayaAgent agent, List<EvalCase> cases, int batchSize)
    {
        var scored = new List<ScoredQuestion>();
        var dropped = 0;
        foreach (var group in cases.Index().GroupBy(c => c.Item.QuestionsKey, StringComparer.Ordinal))
        {
            var items = group.ToList();
            var questions = items[0].Item.Questions;
            IReadOnlyList<OrderedDictionary<string, float[]>> logits;
            try
            {
                logits = agent.PredictLogits(items.Select(c => c.Item.State), questions, new BatchOptions { BatchSize = batchSize, SortByLength = true });
            }
            catch (ArgumentException e)
            {
                // The harness drops sequences whose options don't fit head_max_len; so do we.
                Console.Error.WriteLine($"  dropped {items.Count} cases: {e.Message}");
                dropped += items.Count;
                continue;
            }
            for (var i = 0; i < items.Count; i++)
            {
                var (index, c) = items[i];
                foreach (var (id, z) in logits[i])
                    if (c.Gold.TryGetValue(id, out var gold))
                        scored.Add(new ScoredQuestion(index, id, questions[id].Type, z, gold, c.Workflow));
            }
        }
        return (scored.OrderBy(s => s.Case).ToList(), dropped);
    }

    /// <summary>Metrics at the agent's own (shipped) temperatures.</summary>
    public static ClassificationReport Shipped(LayaAgent agent, IEnumerable<ScoredQuestion> scored) =>
        CalibrationMetrics.Evaluate(scored.Select(s => s.At(agent.Temperatures.For(s.Type, s.Logits.Length))));
}
