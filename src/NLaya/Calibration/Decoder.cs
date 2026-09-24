using System.Globalization;
using System.Numerics.Tensors;
using NLaya.Backends;

namespace NLaya.Calibration;

/// <summary>Turns logit rows into typed answers. Port of <c>Agent._decode_answers</c>.</summary>
internal static class Decoder
{
    public static OrderedMap<Answer> Decode(BackendOutput output, float[] actProbs, int rowOffset,
        IReadOnlyList<string> questionIds, IReadOnlyList<Question> questions, IReadOnlyList<int> optionCounts,
        TemperatureTable temps, string? lang)
    {
        var answers = new OrderedMap<Answer>();
        for (var j = 0; j < questionIds.Count; j++)
        {
            var r = rowOffset + j;
            var q = questions[j];
            var k = optionCounts[j];
            var t = temps.For(q.Type, k, lang);

            // float32 throughout, like the numpy code it mirrors.
            var p = new float[k];
            TensorPrimitives.Divide(output.Logits.AsSpan(r * output.MaxMarkers, k), (float)t, p);
            StableSoftmax(p, p);

            var ansConf = Round(AnswerConfidence(p, k));
            var action = new ActionInfo(Round(actProbs[r * output.ActOutputs]));

            switch (q.Type)
            {
                case QuestionType.Choice:
                {
                    var probs = new OrderedMap<double>();
                    for (var i = 0; i < k; i++) probs[q.Options[i].Key] = Round(p[i]);
                    answers[questionIds[j]] = new ChoiceAnswer
                    {
                        Choice = q.Options[ArgMax(p)].Key,
                        Probabilities = probs,
                        Confidence = Round(EntropyConfidence(p, k)),
                        AnswerConfidence = ansConf,
                        Action = action,
                    };
                    break;
                }
                case QuestionType.Score:
                {
                    var probs = new OrderedMap<double>();
                    double expected = 0;
                    for (var i = 0; i < k; i++)
                    {
                        probs[i.ToString(CultureInfo.InvariantCulture)] = Round(p[i]);
                        expected += i * (double)p[i];
                    }
                    answers[questionIds[j]] = new ScoreAnswer
                    {
                        Score = Round(expected),
                        Legend = q.Levels.Select(l => l?.DeepClone()).ToList(),
                        Probabilities = probs,
                        Confidence = Round(EntropyConfidence(p, k)),
                        AnswerConfidence = ansConf,
                        Action = action,
                    };
                    break;
                }
                default:
                {
                    double pTrue = p[1];
                    answers[questionIds[j]] = new NoulAnswer
                    {
                        Noul = Round(pTrue),
                        Confidence = Round(Math.Max(pTrue, 1.0 - pTrue)),
                        AnswerConfidence = ansConf,
                        Action = action,
                    };
                    break;
                }
            }
        }
        return answers;
    }

    public static float[] Softmax(float[] logits, int rows, int cols)
    {
        var o = new float[logits.Length];
        for (var r = 0; r < rows; r++) StableSoftmax(logits.AsSpan(r * cols, cols), o.AsSpan(r * cols, cols));
        return o;
    }

    /// <summary>
    /// <see cref="TensorPrimitives.SoftMax(ReadOnlySpan{float}, Span{float})"/> exponentiates the raw
    /// values, which overflows for the act head's ~1e3 logits; shift by the max first, as numpy/torch do.
    /// </summary>
    private static void StableSoftmax(ReadOnlySpan<float> x, Span<float> dest)
    {
        TensorPrimitives.Subtract(x, TensorPrimitives.Max(x), dest);
        TensorPrimitives.SoftMax(dest, dest);
    }

    /// <summary>max(p): the quantity temperature scaling fits (Python <c>answer_confidence</c>).</summary>
    public static double AnswerConfidence(float[] p, int k) => k < 1 ? 1.0 : Math.Clamp(TensorPrimitives.Max(p.AsSpan(0, k)), 0.0, 1.0);

    /// <summary>1 - H(p)/log(k) (Python <c>confidence_from_probs</c>).</summary>
    public static double EntropyConfidence(float[] p, int k)
    {
        if (k < 2) return 1.0;
        float ent = 0;
        for (var i = 0; i < k; i++) ent -= p[i] * MathF.Log(Math.Clamp(p[i], 1e-12f, 1.0f));
        return Math.Clamp(1.0f - (float)(ent / Math.Log(k)), 0.0f, 1.0f);
    }

    private static int ArgMax(float[] p) => TensorPrimitives.IndexOfMax(p);

    /// <summary>Python round(x, 4): half-to-even.</summary>
    public static double Round(double x) => Math.Round(x, 4, MidpointRounding.ToEven);
}
