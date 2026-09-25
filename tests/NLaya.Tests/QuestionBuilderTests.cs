namespace NLaya.Tests;

public class QuestionBuilderTests
{
    private const string PythonShape = """
        {
          "team": {"type": "choice", "instructions": "Which team should handle `body`?",
                   "criteria": {"billing": "Payments and refunds", "support": "Product issues", "other": null}},
          "lang": {"type": "choice", "instructions": "Which language?", "criteria": {"en": null, "de": null}},
          "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["routine", "soon", "critical"]},
          "refund": {"type": "noul", "instructions": "Does the customer ask for a refund?"},
          "reply": {"type": "noul", "instructions": "Does the sender expect a reply?",
                    "criteria": {"false": "an FYI", "true": "they asked a question"}, "labels": {"false": "no", "true": "yes"}}
        }
        """;

    [Fact]
    public void Fluent_chain_matches_the_python_dict_shape()
    {
        var fluent = new Questions()
            .Choice("team", "Which team should handle `body`?",
                ("billing", "Payments and refunds"),
                ("support", "Product issues"),
                ("other", null))
            .Choice("lang", "Which language?", "en", "de")
            .Score("urgency", "How urgent is this?", "routine", "soon", "critical")
            .Noul("refund", "Does the customer ask for a refund?")
            .Noul("reply", "Does the sender expect a reply?", n => n
                .WhenFalse("an FYI")
                .WhenTrue("they asked a question")
                .Labels("no", "yes"));

        JsonAssert.Equivalent(Questions.Parse(PythonShape).ToJson(), fluent.ToJson());
    }

    [Fact]
    public void Nested_builders_match_the_params_forms()
    {
        var nested = new Questions()
            .Choice("team", "Which team should handle `body`?", c => c
                .Option("billing", "Payments and refunds")
                .Option("support", "Product issues")
                .Option("other"))
            .Choice("lang", "Which language?", c =>
            {
                foreach (var l in new[] { "en", "de" }) c.Option(l);
            })
            .Score("urgency", "How urgent is this?", s => s.Level("routine").Level("soon").Level("critical"));

        var expected = Questions.Parse(PythonShape);
        JsonAssert.Equivalent(expected["team"].ToJson(), nested["team"].ToJson());
        JsonAssert.Equivalent(expected["lang"].ToJson(), nested["lang"].ToJson());
        JsonAssert.Equivalent(expected["urgency"].ToJson(), nested["urgency"].ToJson());
    }

    [Fact]
    public void Fluent_and_indexer_forms_are_interchangeable()
    {
        var q = new Questions { ["refund"] = Question.Noul("Refund?") }
            .With("team", Question.Choice("Team?", ("billing", "money"), ("other", null)))
            .Noul("urgent", "Urgent?");

        Assert.Equal(["refund", "team", "urgent"], q.Keys);
        Assert.Equal(QuestionType.Choice, q["team"].Type);
    }

    [Fact]
    public void Tuple_overload_matches_the_dictionary_overload()
    {
        var tuples = Question.Choice("Team?", ("billing", "money"), ("other", null));
        var dict = Question.Choice("Team?", new Dictionary<string, string?> { ["billing"] = "money", ["other"] = null });
        JsonAssert.Equivalent(dict.ToJson(), tuples.ToJson());
    }

    [Fact]
    public void Bare_labels_still_bind_to_the_string_overload()
    {
        var q = Question.Choice("Team?", "billing", "support");
        Assert.All(q.Options, o => Assert.Null(o.Value));
        Assert.Equal(["billing", "support"], q.Options.Select(o => o.Key));
    }

    [Fact]
    public void Repeated_id_in_a_chain_throws()
    {
        var q = new Questions().Noul("refund", "Refund?");
        var e = Assert.Throws<ArgumentException>(() => q.Noul("refund", "Again?"));
        Assert.Contains("'refund' is already defined", e.Message);
    }

    [Fact]
    public void Indexer_still_replaces_like_a_python_dict()
    {
        var q = new Questions().Noul("refund", "Refund?");
        q["refund"] = Question.Noul("Again?");
        Assert.Equal("Again?", q["refund"].InstructionText);
    }

    [Fact]
    public void Builders_share_the_factory_validation()
    {
        Assert.Contains("at least one criterion",
            Assert.Throws<ArgumentException>(() => new Questions().Choice("team", "Team?", _ => { })).Message);
        Assert.Contains("labels must be distinct",
            Assert.Throws<ArgumentException>(() => new Questions().Choice("team", "Team?", c => c.Option("a").Option("a"))).Message);
        Assert.Contains("at least one level",
            Assert.Throws<ArgumentException>(() => new Questions().Score("urgency", "Urgent?", _ => { })).Message);
        Assert.Contains("labels must be distinct",
            Assert.Throws<ArgumentException>(() => new Questions().Choice("team", "Team?", ("a", "x"), ("a", "y"))).Message);
    }

    [Fact]
    public void A_failed_add_leaves_the_set_unchanged()
    {
        var q = new Questions().Noul("refund", "Refund?");
        Assert.Throws<ArgumentException>(() => q.Score("urgency", "Urgent?", _ => { }));
        Assert.Equal(["refund"], q.Keys);
    }

    [Fact]
    public void TryGet_finds_answers_by_id_and_type()
    {
        var answers = new OrderedDictionary<string, Answer>
        {
            ["refund"] = new NoulAnswer { Noul = 0.9 },
        };
        var result = new LayaResult(LayaAgent.ResultModelName, answers, Usage.Zero);

        Assert.True(result.TryGet("refund", out var a));
        Assert.IsType<NoulAnswer>(a);
        Assert.True(result.TryGet<NoulAnswer>("refund", out var noul));
        Assert.Equal(0.9, noul.Noul);
        Assert.False(result.TryGet<ChoiceAnswer>("refund", out var choice));
        Assert.Null(choice);
        Assert.False(result.TryGet("missing", out _));
    }
}
