using Microsoft.ML;
using Microsoft.ML.Data;

using NLaya.ML;

namespace NLaya.Tests.ML;

public class LayaTransformerTests
{
    private static readonly Questions Questions = new Questions()
        .Choice("team", "Which team?", "billing", "support")
        .Score("urgency", "How urgent?", "low", "medium", "high")
        .Noul("refund", "Asks for a refund");

    private readonly MLContext _ml = new(seed: 0);

    private IDataView Tickets(int n) => _ml.Data.LoadFromEnumerable(Enumerable.Range(0, n).Select(i => new Ticket
    {
        Body = i % 3 == 0 ? $"urgent: ticket {i}" : $"ticket {i}",
        Id = i,
        Amount = i / 10f,
    }));

    [Fact]
    public void Adds_typed_columns_with_slot_names()
    {
        var schema = _ml.Transforms.Laya(new TextPredictor(), Questions, "Body").Fit(Tickets(1)).Transform(Tickets(1)).Schema;

        Assert.Equal(["Body", "Id", "Amount", "team", "team_probs", "team_confidence", "urgency", "urgency_probs", "urgency_confidence",
            "refund", "refund_probability", "refund_confidence"], schema.Select(c => c.Name));
        Assert.Equal(TextDataViewType.Instance, schema["team"].Type);
        Assert.Equal(new VectorDataViewType(NumberDataViewType.Single, 2), schema["team_probs"].Type);
        Assert.Equal(NumberDataViewType.Single, schema["urgency"].Type);
        Assert.Equal(BooleanDataViewType.Instance, schema["refund"].Type);

        VBuffer<ReadOnlyMemory<char>> slots = default;
        schema["team_probs"].GetSlotNames(ref slots);
        Assert.Equal(["billing", "support"], slots.DenseValues().Select(s => s.ToString()));
        schema["urgency_probs"].GetSlotNames(ref slots);
        Assert.Equal(["0", "1", "2"], slots.DenseValues().Select(s => s.ToString()));
    }

    [Fact]
    public void Answers_line_up_with_their_rows_across_chunks()
    {
        var predictor = new TextPredictor();
        var options = new LayaTransformerOptions { ChunkSize = 3 };
        var scored = _ml.Transforms.Laya(predictor, Questions, options, "Body").Fit(Tickets(7)).Transform(Tickets(7));

        var ids = scored.GetColumn<int>("Id").ToArray();
        var teams = scored.GetColumn<string>("team").ToArray();
        var refunds = scored.GetColumn<bool>("refund").ToArray();
        var probs = scored.GetColumn<float[]>("urgency_probs").ToArray();

        Assert.Equal(Enumerable.Range(0, 7), ids);
        Assert.Equal(ids.Select(i => i % 3 == 0 ? "support" : "billing"), teams);
        Assert.Equal(ids.Select(i => i % 3 == 0), refunds);
        Assert.Equal([0.1f, 0.1f, 0.8f], probs[0]);
        // Three columns were read, each through its own cursor: 3 + 3 + 1 rows per pass.
        Assert.Equal([3, 3, 1, 3, 3, 1, 3, 3, 1], predictor.Batches.Select(b => b.Count));
    }

    [Fact]
    public void Rows_read_back_into_classes()
    {
        var scored = _ml.Transforms.Laya(new TextPredictor(), Questions, "Body").Fit(Tickets(2)).Transform(Tickets(2));
        var rows = _ml.Data.CreateEnumerable<ScoredTicket>(scored, reuseRowObject: false).ToList();

        Assert.Equal(["support", "billing"], rows.Select(r => r.Team));
        Assert.Equal([0.2f, 0.8f], rows[0].TeamProbabilities, (a, b) => Math.Abs(a - b) < 1e-6);
        Assert.Equal([true, false], rows.Select(r => r.Refund));
    }

    [Fact]
    public void A_cached_view_answers_each_row_once()
    {
        var predictor = new TextPredictor();
        var transformed = _ml.Transforms.Laya(predictor, Questions, "Body").Fit(Tickets(4)).Transform(Tickets(4));
        // Prefetching every column fills the cache in one pass; without it, each column fills on first read.
        var scored = _ml.Data.Cache(transformed, [.. transformed.Schema.Select(c => c.Name)]);

        _ = scored.GetColumn<string>("team").ToArray();
        _ = scored.GetColumn<bool>("refund").ToArray();
        Assert.Equal(4, predictor.Batches.Sum(b => b.Count));
    }

    [Fact]
    public void Reading_only_input_columns_runs_no_inference()
    {
        var predictor = new TextPredictor();
        var scored = _ml.Transforms.Laya(predictor, Questions, "Body").Fit(Tickets(5)).Transform(Tickets(5));

        Assert.Equal(5, scored.GetColumn<int>("Id").Count());
        Assert.Empty(predictor.Batches);
    }

    [Fact]
    public void Several_input_columns_make_a_json_state()
    {
        var predictor = new TextPredictor();
        var scored = _ml.Transforms.Laya(predictor, Questions, "Body", "Id", "Amount").Fit(Tickets(2)).Transform(Tickets(2));
        _ = scored.GetColumn<string>("team").ToArray();

        Assert.Equal(["{\"Body\": \"urgent: ticket 0\", \"Id\": 0, \"Amount\": 0.0}", "{\"Body\": \"ticket 1\", \"Id\": 1, \"Amount\": 0.1}"],
            predictor.Batches.Single());
    }

    [Fact]
    public void Estimator_describes_its_output_and_checks_inputs()
    {
        var estimator = _ml.Transforms.Laya(new TextPredictor(), Questions, "Body");
        var shape = estimator.GetOutputSchema(SchemaShapes.From(Tickets(1).Schema));
        var team = shape.Single(c => c.Name == "team_probs");
        Assert.Equal(SchemaShape.Column.VectorKind.Vector, team.Kind);
        Assert.Equal(NumberDataViewType.Single, team.ItemType);
        Assert.Contains(shape, c => c.Name == "Body");

        var missing = _ml.Transforms.Laya(new TextPredictor(), Questions, "Subject");
        Assert.Throws<ArgumentOutOfRangeException>(() => missing.GetOutputSchema(SchemaShapes.From(Tickets(1).Schema)));
        Assert.Throws<ArgumentOutOfRangeException>(() => missing.Fit(Tickets(1)));
    }

    [Fact]
    public void Saving_is_not_supported()
    {
        var model = _ml.Transforms.Laya(new TextPredictor(), Questions, "Body").Fit(Tickets(1));
        using var stream = new MemoryStream();
        Assert.Throws<NotSupportedException>(() => _ml.Model.Save(model, Tickets(1).Schema, stream));
    }
}
