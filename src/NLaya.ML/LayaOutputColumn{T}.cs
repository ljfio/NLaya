using Microsoft.ML;
using Microsoft.ML.Data;

namespace NLaya.ML;

/// <summary>A <see cref="LayaOutputColumn"/> whose values are <typeparamref name="T"/>.</summary>
internal sealed class LayaOutputColumn<T>(string name, string questionId, DataViewType type, Func<Answer, T> read,
    DataViewSchema.Annotations? annotations = null) : LayaOutputColumn(name, questionId, type, annotations)
{
    public override Delegate CreateGetter(Func<LayaResult> current) =>
        (ValueGetter<T>)((ref T value) =>
        {
            var result = current();
            value = result.TryGet(QuestionId, out var answer)
                ? read(answer)
                : throw new InvalidOperationException($"the answers have no '{QuestionId}' (a hook may have removed it)");
        });
}
