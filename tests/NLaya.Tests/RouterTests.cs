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
        Assert.Equal(c["route"]!["model"]!.GetValue<string>(), d.Model);
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
            var d = Router.Route("Some English text for the router to consider", null, new RouteOptions
            {
                Model = S("model"),
                Task = S("task"),
                Lang = S("lang"),
                LangGuess = guess is null ? null : _ => guess,
            });
            Assert.Equal(c["model"]!.GetValue<string>(), d.Model);
            Assert.Equal(c["reason"]!.GetValue<string>(), d.Reason);
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
        Assert.Equal((w["model"]!.GetValue<string>(), w["reason"]!.GetValue<string>()), (auto.Model, auto.Reason));
        Assert.Equal("english", Router.Route("hello there my friend", qs).Model);
    }

    [Theory]
    [InlineData("en", "english")]
    [InlineData("ML", "multilingual")]
    [InlineData("typed_decisions", "typed-decisions")]
    [InlineData(" laya-multilingual ", "multilingual")]
    public void Normalises_aliases(string alias, string expected) => Assert.Equal(expected, Router.NormaliseName(alias));

    [Fact]
    public void Standalone_repos()
    {
        var r = new Router(new RouterOptions { StandaloneRepos = true });
        Assert.Equal("convaiinnovations/laya-multilingual", r.Route("二重に請求されました").Repo);
        Assert.Equal("convaiinnovations/laya/multilingual", Router.Route("二重に請求されました").Repo);
    }
}
