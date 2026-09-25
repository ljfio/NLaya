using Microsoft.ML;
using Microsoft.ML.Data;

namespace NLaya.ML;

/// <summary>
/// Walks the source row by row for pass-through columns, while a second cursor over the same source
/// reads the input columns a chunk ahead and answers each chunk with one batch call. Source cursors
/// without a random source visit rows in the same order, so the two stay aligned.
/// </summary>
internal sealed class LayaRowCursor : DataViewRowCursor
{
    private readonly LayaDataView _view;
    private readonly DataViewRowCursor _source;
    private readonly DataViewRowCursor? _input;
    private readonly LayaStateReader? _reader;
    private readonly bool[] _active;
    private IReadOnlyList<LayaResult> _chunk = [];
    private int _next;
    private LayaResult? _current;

    public LayaRowCursor(LayaDataView view, DataViewRowCursor source, DataViewRowCursor? input, bool[] active)
    {
        _view = view;
        _source = source;
        _input = input;
        _active = active;
        if (input is not null) _reader = new LayaStateReader(input, view.InputColumns.Select(c => input.Schema[c.Index]).ToList());
    }

    public override DataViewSchema Schema => _view.Schema;
    public override long Position => _source.Position;
    public override long Batch => _source.Batch;

    public override bool MoveNext()
    {
        if (!_source.MoveNext())
        {
            _current = null;
            return false;
        }
        if (_input is not null)
        {
            if (_next == _chunk.Count) ReadChunk();
            _current = _chunk[_next++];
        }
        return true;
    }

    private void ReadChunk()
    {
        var states = new List<LayaState>(_view.Options.ChunkSize);
        while (states.Count < _view.Options.ChunkSize && _input!.MoveNext()) states.Add(_reader!.Read());
        if (states.Count == 0) throw new InvalidOperationException("the source returned fewer rows on a second cursor; Laya needs a source that repeats its rows in order");
        _chunk = _view.Predictor.PredictBatch(states, _view.Questions, _view.Options.Batch);
        if (_chunk.Count != states.Count)
            throw new InvalidOperationException($"the predictor returned {_chunk.Count} results for {states.Count} states");
        _next = 0;
    }

    private LayaResult Current => _current ?? throw new InvalidOperationException("the cursor is not on a row; call MoveNext first");

    public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
    {
        if (!IsColumnActive(column)) throw new InvalidOperationException($"column '{column.Name}' was not requested when the cursor was created");
        var sourceCount = _view.Source.Schema.Count;
        if (column.Index < sourceCount) return _source.GetGetter<TValue>(_source.Schema[column.Index]);
        return _view.OutputColumns[column.Index - sourceCount].CreateGetter(() => Current) as ValueGetter<TValue>
            ?? throw new InvalidOperationException($"column '{column.Name}' is {column.Type}, not {typeof(TValue).Name}");
    }

    public override ValueGetter<DataViewRowId> GetIdGetter() => _source.GetIdGetter();

    public override bool IsColumnActive(DataViewSchema.Column column) => _active[column.Index];

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _source.Dispose();
            _input?.Dispose();
        }
        base.Dispose(disposing);
    }
}
