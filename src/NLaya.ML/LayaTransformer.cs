using Microsoft.ML;
using Microsoft.ML.Data;

namespace NLaya.ML;

/// <summary>
/// Adds Laya's answers to an <see cref="IDataView"/> as columns (see <see cref="LayaEstimator"/> for
/// their names and types). Transforming is lazy: rows are answered, a chunk at a time, as a consumer
/// reads the answer columns.
/// </summary>
/// <remarks>
/// Saving isn't supported: the model lives in the <see cref="ILayaPredictor"/>, not in the ML.NET
/// model file. Rebuild the pipeline with the predictor instead. There's no row-to-row mapper either,
/// so no <c>PredictionEngine</c>; for single predictions call the predictor directly.
/// </remarks>
public sealed class LayaTransformer : ITransformer
{
    internal LayaTransformer(ILayaPredictor predictor, Questions questions, IReadOnlyList<string> inputColumnNames, LayaTransformerOptions options)
    {
        Predictor = predictor;
        Questions = questions;
        InputColumnNames = inputColumnNames;
        Options = options;
    }

    public ILayaPredictor Predictor { get; }
    public Questions Questions { get; }
    public IReadOnlyList<string> InputColumnNames { get; }
    public LayaTransformerOptions Options { get; }

    public bool IsRowToRowMapper => false;

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
    {
        ArgumentNullException.ThrowIfNull(inputSchema);
        LayaStateReader.Resolve(inputSchema, InputColumnNames);
        return BuildSchema(inputSchema, LayaOutputColumn.For(Questions, Options.OutputColumnPrefix));
    }

    public IDataView Transform(IDataView input)
    {
        ArgumentNullException.ThrowIfNull(input);
        return new LayaDataView(input, Predictor, Questions, InputColumnNames, Options);
    }

    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema) =>
        throw new NotSupportedException("LayaTransformer answers rows in batches and has no row-to-row mapper; call the ILayaPredictor directly for single rows.");

    void ICanSaveModel.Save(ModelSaveContext ctx) =>
        throw new NotSupportedException("LayaTransformer can't be saved: the Laya model isn't part of the ML.NET model. Rebuild the pipeline with the predictor after loading.");

    internal static DataViewSchema BuildSchema(DataViewSchema input, IEnumerable<LayaOutputColumn> outputs)
    {
        var builder = new DataViewSchema.Builder();
        builder.AddColumns(input);
        foreach (var c in outputs) builder.AddColumn(c.Name, c.Type, c.Annotations);
        return builder.ToSchema();
    }
}
