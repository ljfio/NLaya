using Microsoft.ML;
using Microsoft.ML.Data;

namespace NLaya.Parity;

/// <summary>The ML.NET stage on a real checkpoint: its columns hold exactly what PredictBatch returns.</summary>
[Collection(ParityCollection.Name)]
public class MLTransformerParityTests(ParityFixture fx)
{
    private sealed class Row
    {
        public string Text { get; set; } = "";
    }

    [Fact]
    public void Transformer_columns_equal_PredictBatch()
    {
        var agent = fx.Agent("torchsharp", "multilingual");
        var texts = new[]
        {
            "I was charged twice this month, please refund me.",
            "Die App stürzt beim Start ab, bitte dringend helfen!",
            "What time does the office open?",
            "No consigo iniciar sesión desde ayer.",
            "Cancel my subscription now or I'm calling my bank.",
        };
        var questions = Presets.Triage();
        var ml = new MLContext(seed: 0);
        var data = ml.Data.LoadFromEnumerable(texts.Select(t => new Row { Text = t }));

        var transformed = ml.Transforms.Laya(agent, questions, new NLaya.ML.LayaTransformerOptions { ChunkSize = 2 }, "Text")
            .Fit(data).Transform(data);
        var scored = ml.Data.Cache(transformed, [.. transformed.Schema.Select(c => c.Name)]);
        var expected = agent.PredictBatch(texts.Select(t => (LayaState)t), questions, new BatchOptions { BatchSize = 32, SortByLength = true });

        foreach (var (id, q) in questions)
        {
            var confidence = scored.GetColumn<float>(id + "_confidence").ToArray();
            for (var i = 0; i < texts.Length; i++)
                Assert.Equal((float)expected[i][id].AnswerConfidence, confidence[i]);
            switch (q.Type)
            {
                case QuestionType.Choice:
                    Assert.Equal(expected.Select(r => r.Answer<ChoiceAnswer>(id).Choice), scored.GetColumn<string>(id));
                    break;
                case QuestionType.Score:
                    Assert.Equal(expected.Select(r => (float)r.Answer<ScoreAnswer>(id).Score), scored.GetColumn<float>(id));
                    break;
                default:
                    Assert.Equal(expected.Select(r => (float)r.Answer<NoulAnswer>(id).Probability), scored.GetColumn<float>(id + "_probability"));
                    break;
            }
        }
    }
}
