using NLaya.Routing;

namespace NLaya.Tests;

public class RouterTests
{
    private static readonly Router Router = new(_ => throw new InvalidOperationException("routing tests never load a model"));

    public static TheoryData<int> Cases() => new(Enumerable.Range(0, TestFiles.Fixture("lang.json")["cases"]!.AsArray().Count));

    [Theory]
    [MemberData(nameof(Cases))]
    public void Routes_like_python(int i)
    {
        var c = TestFiles.Fixture("lang.json")["cases"]![i]!;
        var d = Router.Route(LayaState.FromJson(c["state"]?.DeepClone()));
        Assert.Equal(c["route"]!["model"]!.GetValue<string>(), d.Checkpoint.Name());
        Assert.Equal(c["route"]!["repo"]!.GetValue<string>(), d.Repo);
        Assert.Equal(c["route"]!["reason"]!.GetValue<string>(), d.Reason);
    }

    [Fact]
    public void Explicit_options_take_precedence_like_python()
    {
        foreach (var c in TestFiles.Fixture("lang.json")["explicit"]!.AsArray())
        {
            var kw = c!["kwargs"]!.AsObject();
            string? S(string k) => kw[k]?.GetValue<string>();
            var guess = S("lang_guess");
            // Python's model= and task= strings are one typed Checkpoint in .NET, so the reason names the
            // checkpoint rather than echoing the alias the caller typed.
            var pinned = (S("model") ?? S("task")) is { } name ? Checkpoints.Parse(name) : (Checkpoint?)null;
            var d = Router.Route("Some English text for the router to consider", null, new RouteOptions
            {
                Checkpoint = pinned,
                Lang = S("lang"),
                LangGuess = guess is null ? null : _ => guess,
            });
            Assert.Equal(c["model"]!.GetValue<string>(), d.Checkpoint.Name());
            Assert.Equal(pinned is { } p ? $"explicit model='{p.Name()}'" : c["reason"]!.GetValue<string>(), d.Reason);
        }
    }

    [Fact]
    public void Typed_decisions_workflow_needs_opt_in()
    {
        var w = TestFiles.Fixture("lang.json")["workflow"]!;
        var ids = w["questions"]!.AsArray().Select(x => x!.GetValue<string>()).ToList();
        var qs = new Questions();
        foreach (var id in ids) qs[id] = Question.Noul("x");

        Assert.Equal(w["workflow"]!.GetValue<string>(), Router.MatchTypedDecisionsWorkflow(ids));
        var auto = new Router(new RouterOptions { AutoTaskDetection = true }).Route("hello there my friend", qs);
        Assert.Equal((w["model"]!.GetValue<string>(), w["reason"]!.GetValue<string>()), (auto.Checkpoint.Name(), auto.Reason));
        Assert.Equal(Checkpoint.English, Router.Route("hello there my friend", qs).Checkpoint);
    }

    [Theory]
    [InlineData("en", Checkpoint.English)]
    [InlineData("ML", Checkpoint.Multilingual)]
    [InlineData("typed_decisions", Checkpoint.TypedDecisions)]
    [InlineData(" laya-multilingual ", Checkpoint.Multilingual)]
    [InlineData("TypedDecisions", Checkpoint.TypedDecisions)]
    public void Parses_names_and_aliases(string alias, Checkpoint expected) => Assert.Equal(expected, Checkpoints.Parse(alias));

    [Fact]
    public void Names_are_pythons() =>
        Assert.Equal(["english", "multilingual", "typed-decisions"], Enum.GetValues<Checkpoint>().Select(c => c.Name()));

    [Fact]
    public void Unknown_names_list_the_choices() =>
        Assert.StartsWith("unknown model 'nope'; choose one of ['english', 'multilingual', 'typed-decisions']",
            Assert.Throws<ArgumentException>(() => Checkpoints.Parse("nope")).Message);

    [Fact]
    public void Standalone_repos()
    {
        var r = new Router(new RouterOptions { StandaloneRepos = true });
        Assert.Equal("convaiinnovations/laya-multilingual", r.Route("二重に請求されました").Repo);
        Assert.Equal("convaiinnovations/laya/multilingual", Router.Route("二重に請求されました").Repo);
    }
}
