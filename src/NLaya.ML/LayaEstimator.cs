using Microsoft.ML;

namespace NLaya.ML;

/// <summary>
/// An ML.NET pipeline step that answers Laya questions about each row. Nothing is trained:
/// <see cref="Fit"/> only checks the input columns. For each question id it adds:
/// <list type="bullet">
/// <item><b>choice</b>: <c>id</c> (text, the label) and <c>id_probs</c> (one slot per option, named by label).</item>
/// <item><b>score</b>: <c>id</c> (single, the expected score) and <c>id_probs</c> (one slot per level).</item>
/// <item><b>noul</b>: <c>id</c> (boolean) and <c>id_probability</c> (single, P(true)).</item>
/// <item>every question: <c>id_confidence</c> (single, the calibrated probability of the answer).</item>
/// </list>
/// One text column is the state's text; several columns become a JSON object keyed by column name.
/// </summary>
public sealed class LayaEstimator : IEstimator<LayaTransformer>
{
    private readonly LayaTransformer _transformer;

    internal LayaEstimator(ILayaPredictor predictor, Questions questions, IReadOnlyList<string> inputColumnNames, LayaTransformerOptions options)
    {
        ArgumentNullException.ThrowIfNull(predictor);
        ArgumentNullException.ThrowIfNull(questions);
        ArgumentOutOfRangeException.ThrowIfLessThan(options.ChunkSize, 1);
        if (questions.Count == 0) throw new ArgumentException("ask at least one question", nameof(questions));
        _transformer = new LayaTransformer(predictor, questions, inputColumnNames, options);
    }

    public LayaTransformer Fit(IDataView input)
    {
        ArgumentNullException.ThrowIfNull(input);
        _transformer.GetOutputSchema(input.Schema);
        return _transformer;
    }

    public SchemaShape GetOutputSchema(SchemaShape inputSchema)
    {
        ArgumentNullException.ThrowIfNull(inputSchema);
        foreach (var name in _transformer.InputColumnNames)
        {
            // Like DataViewSchema, a later column hides an earlier one with the same name.
            var c = inputSchema.LastOrDefault(c => c.Name == name);
            if (c.Name is null) throw new ArgumentOutOfRangeException(nameof(inputSchema), $"input column '{name}' not found");
            if (c.Kind != SchemaShape.Column.VectorKind.Scalar || !LayaStateReader.IsSupported(c.ItemType))
                throw new ArgumentOutOfRangeException(nameof(inputSchema), $"input column '{name}' must be a text, number or boolean scalar");
        }
        var outputs = SchemaShapes.From(LayaTransformer.BuildSchema(new DataViewSchema.Builder().ToSchema(),
            LayaOutputColumn.For(_transformer.Questions, _transformer.Options.OutputColumnPrefix)));
        var names = outputs.Select(c => c.Name).ToHashSet();
        return new SchemaShape(inputSchema.Where(c => !names.Contains(c.Name)).Concat(outputs));
    }
}
