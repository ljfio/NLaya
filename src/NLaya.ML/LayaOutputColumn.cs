using Microsoft.ML;
using Microsoft.ML.Data;

namespace NLaya.ML;

/// <summary>One column <see cref="LayaTransformer"/> adds, read from the answer to one question.</summary>
internal abstract class LayaOutputColumn(string name, string questionId, DataViewType type, DataViewSchema.Annotations? annotations)
{
    public string Name { get; } = name;
    public string QuestionId { get; } = questionId;
    public DataViewType Type { get; } = type;
    public DataViewSchema.Annotations? Annotations { get; } = annotations;

    /// <summary>A <c>ValueGetter&lt;T&gt;</c> over the answer of the row <paramref name="current"/> returns.</summary>
    public abstract Delegate CreateGetter(Func<LayaResult> current);

    /// <summary>
    /// The columns for each question: its answer under the question id (the label, expected score
    /// or yes/no), <c>_probs</c> (one slot per option, named) or <c>_probability</c> (P(true)), and
    /// <c>_confidence</c> (the calibrated max(p)).
    /// </summary>
    public static List<LayaOutputColumn> For(Questions questions, string prefix)
    {
        var columns = new List<LayaOutputColumn>();
        foreach (var (id, q) in questions)
        {
            var name = prefix + id;
            switch (q.Type)
            {
                case QuestionType.Choice:
                {
                    var labels = q.Options.Select(o => o.Key).ToArray();
                    columns.Add(new LayaOutputColumn<ReadOnlyMemory<char>>(name, id, TextDataViewType.Instance,
                        a => ((ChoiceAnswer)a).Choice.AsMemory()));
                    columns.Add(Probabilities(name, id, labels, a => ((ChoiceAnswer)a).Probabilities));
                    break;
                }
                case QuestionType.Score:
                {
                    var levels = Enumerable.Range(0, q.Levels.Count).Select(i => i.ToString(System.Globalization.CultureInfo.InvariantCulture)).ToArray();
                    columns.Add(new LayaOutputColumn<float>(name, id, NumberDataViewType.Single, a => (float)((ScoreAnswer)a).Score));
                    columns.Add(Probabilities(name, id, levels, a => ((ScoreAnswer)a).Probabilities));
                    break;
                }
                default:
                    columns.Add(new LayaOutputColumn<bool>(name, id, BooleanDataViewType.Instance, a => ((NoulAnswer)a).Value));
                    columns.Add(new LayaOutputColumn<float>(name + "_probability", id, NumberDataViewType.Single, a => (float)((NoulAnswer)a).Noul));
                    break;
            }
            columns.Add(new LayaOutputColumn<float>(name + "_confidence", id, NumberDataViewType.Single, a => (float)a.AnswerConfidence));
        }
        return columns;
    }

    private static LayaOutputColumn<VBuffer<float>> Probabilities(string name, string id, string[] slots,
        Func<Answer, OrderedDictionary<string, double>> probabilities)
    {
        var slotNames = slots.Select(s => s.AsMemory()).ToArray();
        var annotations = new DataViewSchema.Annotations.Builder();
        // "SlotNames" is ML.NET's annotation kind for naming vector slots (AnnotationUtils.Kinds.SlotNames).
        annotations.Add("SlotNames", new VectorDataViewType(TextDataViewType.Instance, slots.Length),
            (ref VBuffer<ReadOnlyMemory<char>> v) => v = new VBuffer<ReadOnlyMemory<char>>(slotNames.Length, slotNames));
        return new LayaOutputColumn<VBuffer<float>>(name + "_probs", id, new VectorDataViewType(NumberDataViewType.Single, slots.Length),
            a =>
            {
                var p = probabilities(a);
                var values = new float[slots.Length];
                for (var i = 0; i < slots.Length; i++) values[i] = p.TryGetValue(slots[i], out var x) ? (float)x : 0f;
                return new VBuffer<float>(values.Length, values);
            },
            annotations.ToAnnotations());
    }
}
