using Microsoft.ML;

namespace NLaya.ML;

/// <summary>The lazy output of <see cref="LayaTransformer.Transform"/>: the source's columns plus the answer columns.</summary>
internal sealed class LayaDataView : IDataView
{
    public LayaDataView(IDataView source, ILayaPredictor predictor, Questions questions, IReadOnlyList<string> inputColumnNames,
        LayaTransformerOptions options)
    {
        Source = source;
        Predictor = predictor;
        Questions = questions;
        Options = options;
        InputColumns = LayaStateReader.Resolve(source.Schema, inputColumnNames);
        OutputColumns = LayaOutputColumn.For(questions, options.OutputColumnPrefix);
        Schema = LayaTransformer.BuildSchema(source.Schema, OutputColumns);
    }

    public IDataView Source { get; }
    public ILayaPredictor Predictor { get; }
    public Questions Questions { get; }
    public LayaTransformerOptions Options { get; }
    public DataViewSchema.Column[] InputColumns { get; }
    public List<LayaOutputColumn> OutputColumns { get; }
    public DataViewSchema Schema { get; }

    // Rows are answered in chunks, in source order.
    public bool CanShuffle => false;

    public long? GetRowCount() => Source.GetRowCount();

    public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random? rand = null)
    {
        var active = new bool[Schema.Count];
        foreach (var c in columnsNeeded) active[c.Index] = true;
        var sourceCount = Source.Schema.Count;
        var passThrough = Source.Schema.Where(c => active[c.Index]);
        // Only answer the questions when an answer column is read: counting rows or reading inputs is free.
        var answers = active.Skip(sourceCount).Any(a => a);
        var input = answers ? Source.GetRowCursor(InputColumns) : null;
        return new LayaRowCursor(this, Source.GetRowCursor(passThrough), input, active);
    }

    public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random? rand = null) =>
        [GetRowCursor(columnsNeeded, rand)];
}
