using System.Text.Json.Nodes;

using NLaya.Tests.ExtensionsAI;

namespace NLaya.Tests.Decisions;

/// <summary>C# types to questions (<see cref="DecisionSchema.For{T}(System.Text.Json.Serialization.Metadata.JsonTypeInfo{T})"/>) and the <c>Decide</c> extensions.</summary>
public class DecideTests
{
    private static readonly JsonNode Structured = TestFiles.Fixture("structured.json");

    private static string Questions(DecisionSchema schema) => PythonJson.Serialize(schema.Questions.ToJson());

    [Fact]
    public void A_record_asks_what_the_equivalent_python_schema_asks()
    {
        var python = Structured["schemas"]!.AsArray().Single(c => c!["name"]!.GetValue<string>() == "laya_test_schema")!["questions"]!.AsObject();
        var expected = new JsonObject();
        foreach (var id in new[] { "department", "urgency", "needs_human" }) expected[id] = python[id]!.DeepClone();

        var schema = DecisionSchema.For(DecisionTestContext.Default.LayaTestTicket);

        Assert.Equal(PythonJson.Serialize(expected), Questions(schema));
        Assert.Equal(["department", "urgency", "needs_human"], schema.Fields);
        // No enum member descriptions: the plain enum, exactly as Python would see it.
        Assert.NotNull(schema.ToJsonSchema()["properties"]!["department"]!["enum"]);
        Assert.Equal("object", schema.ToJsonSchema()["type"]!.GetValue<string>());
    }

    [Fact]
    public void Enum_member_descriptions_become_option_descriptions()
    {
        var schema = DecisionSchema.For(DecisionTestContext.Default.TriageTicket);
        var expected = JsonNode.Parse("""
            {
              "Team": {"type": "choice", "instructions": "What is `Team`?",
                       "criteria": {"Billing": "Payments, invoices and refunds", "Support": "Bugs and outages", "other_team": null}},
              "Urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["1", "2", "3"]},
              "NeedsHuman": {"type": "noul", "instructions": "Is `NeedsHuman` true?"},
              "Escalate": {"type": "choice", "instructions": "Who should it be escalated to, if anyone?",
                           "criteria": {"Billing": "Payments, invoices and refunds", "Support": "Bugs and outages", "other_team": null, "null": null}}
            }
            """);
        Assert.Equal(PythonJson.Serialize(expected), Questions(schema));
    }

    [Fact]
    public void Plans_are_cached_per_type_info()
    {
        Assert.Same(DecisionSchema.For(DecisionTestContext.Default.TriageTicket), DecisionSchema.For(DecisionTestContext.Default.TriageTicket));
    }

    [Fact]
    public void Questions_is_a_copy()
    {
        var schema = DecisionSchema.For(DecisionTestContext.Default.TriageTicket);
        schema.Questions.Remove("Team");
        Assert.True(schema.Questions.ContainsKey("Team"));
    }

    [Fact]
    public void A_free_string_property_is_rejected_with_python_wording()
    {
        var ex = Assert.Throws<LayaSchemaException>(() => DecisionSchema.For(DecisionTestContext.Default.FreeTextTicket));
        Assert.Equal("properties.Summary: a free string cannot be a fixed option set; use 'enum' or a boolean", ex.Message);
    }

    [Fact]
    public void Enums_must_serialize_as_strings()
    {
        var ex = Assert.Throws<LayaSchemaException>(() => DecisionSchema.For(NumericEnumContext.Default.LayaTestTicket));
        Assert.Contains("properties.department: enum Department serializes as a number", ex.Message);
    }

    [Fact]
    public void OneOf_consts_describe_options_where_python_would_reject_the_property()
    {
        var schema = DecisionSchema.Parse("""
            {"properties": {
              "team": {"type": "string", "oneOf": [{"const": "billing", "description": "money"}, {"const": "support"}]},
              "flag": {"oneOf": [{"const": true, "description": "it is on"}, {"const": false}]},
              "level": {"type": "integer", "minimum": 0, "maximum": 2, "oneOf": [{"const": 0, "description": "ignored"}]}
            }}
            """);
        var expected = JsonNode.Parse("""
            {
              "team": {"type": "choice", "instructions": "What is `team`?", "criteria": {"billing": "money", "support": null}},
              "flag": {"type": "noul", "instructions": "Is `flag` true?", "criteria": {"true": "it is on"}},
              "level": {"type": "score", "instructions": "Score `level` from 0 to 2", "criteria": ["0", "1", "2"]}
            }
            """);
        Assert.Equal(PythonJson.Serialize(expected), Questions(schema));
    }

    private static FakePredictor TriagePredictor() => new((_, questions) => new OrderedDictionary<string, Answer>
    {
        ["Team"] = FakePredictor.Choice("other_team"),
        ["Urgency"] = new ScoreAnswer
        {
            Score = 1.1,
            Legend = [],
            Confidence = 0.4,
            AnswerConfidence = 0.7,
            Probabilities = [0.1, 0.7, 0.2],
        },
        ["NeedsHuman"] = FakePredictor.Noul(0.8),
        ["Escalate"] = FakePredictor.Choice("null"),
    });

    [Fact]
    public void Decide_returns_the_typed_value()
    {
        var predictor = TriagePredictor();
        var ticket = predictor.Decide("I was charged twice", DecisionTestContext.Default.TriageTicket);

        Assert.Equal(new TriageTicket(Team.Other, 2, true, null), ticket);
        Assert.Equal(["Team", "Urgency", "NeedsHuman", "Escalate"], predictor.Calls.Single().Questions.Keys);
    }

    [Fact]
    public void DecideWithDetails_carries_confidence_and_probabilities()
    {
        var d = TriagePredictor().DecideWithDetails("I was charged twice", DecisionTestContext.Default.TriageTicket);

        Assert.Equal(new TriageTicket(Team.Other, 2, true, null), d.Value);
        Assert.Equal("""{"Team":"other_team","Urgency":2,"NeedsHuman":true,"Escalate":null}""", d.Values.ToJsonString());
        Assert.Equal(0.4, d.Confidence["Urgency"]);
        Assert.Equal(new OrderedDictionary<string, double> { ["false"] = 0.2, ["true"] = 0.8 }, d.Probabilities["NeedsHuman"]);
        Assert.Equal(0.7, d.Probabilities["Urgency"]["1"]);
        Assert.Equal(new Usage(1), d.Usage);
        Assert.Null(d.Routing);
        Assert.Equal(["values", "confidence", "probabilities", "answers", "usage", "routing"], d.ToJson().Select(kv => kv.Key));
    }

    [Fact]
    public async Task DecideAsync_matches_Decide()
    {
        var ticket = await TriagePredictor().DecideAsync("I was charged twice", DecisionTestContext.Default.TriageTicket, ct: TestContext.Current.CancellationToken);
        Assert.Equal(new TriageTicket(Team.Other, 2, true, null), ticket);
    }

    [Fact]
    public void Reflection_overload_reads_the_type_without_a_context()
    {
        var predictor = new FakePredictor((_, _) => new OrderedDictionary<string, Answer>
        {
            ["department"] = FakePredictor.Choice("sales"),
            ["urgency"] = FakePredictor.Score(0),
            ["needs_human"] = FakePredictor.Noul(0.1),
        });
        Assert.Equal(new LayaTestTicket(Department.sales, 0, false), predictor.Decide<LayaTestTicket>("state"));
        Assert.Same(DecisionSchema.For<LayaTestTicket>(), DecisionSchema.For<LayaTestTicket>());
    }

    [Fact]
    public void A_json_schema_decides_json_values()
    {
        var predictor = new FakePredictor((_, _) => new OrderedDictionary<string, Answer>
        {
            ["priority"] = FakePredictor.Choice("2"),
            ["refund"] = FakePredictor.Noul(0.3),
        });
        var schema = JsonNode.Parse("""{"type": "object", "properties": {"priority": {"enum": [1, 2, 3]}, "refund": {"type": "boolean"}}}""")!;

        Assert.Equal("""{"priority":2,"refund":false}""", predictor.Decide("state", schema).ToJsonString());
        var details = predictor.DecideWithDetails("state", schema);
        Assert.Equal(new OrderedDictionary<string, double> { ["false"] = 0.7, ["true"] = 0.3 }, details.Probabilities["refund"]);
    }
}
