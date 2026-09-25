using System.Text.Json.Nodes;

namespace NLaya.Eval;

/// <summary>One line of a suite file: a state, its questions and the correct option of each.</summary>
internal sealed record EvalCase(LayaState State, Questions Questions, string QuestionsKey, Dictionary<string, Gold> Gold, string? Workflow)
{
    public static List<EvalCase> Load(string path, int? limit)
    {
        var cases = new List<EvalCase>();
        foreach (var line in File.ReadLines(path))
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            if (cases.Count == limit) break;
            var o = JsonNode.Parse(line)!.AsObject();
            var questions = o["questions"]!;
            var gold = o["gold"]!.AsObject().ToDictionary(kv => kv.Key, kv => new Gold(
                kv.Value!["index"]!.GetValue<int>(),
                kv.Value["soft"]?.AsArray().Select(x => (float)x!.GetValue<double>()).ToArray()));
            cases.Add(new EvalCase(LayaState.FromJson(o["state"]?.DeepClone()), Questions.FromJson(questions), questions.ToJsonString(),
                gold, o["workflow"]?.GetValue<string>()));
        }
        return cases;
    }
}

/// <summary>The correct option's index, and the teacher's distribution when the suite has one.</summary>
internal sealed record Gold(int Index, float[]? Soft);
