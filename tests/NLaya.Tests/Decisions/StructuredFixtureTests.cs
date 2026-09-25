using System.Text.Json.Nodes;

namespace NLaya.Tests.Decisions;

/// <summary>
/// <see cref="DecisionSchema"/> against <c>laya.structured</c> at the pinned commit
/// (<c>structured.json</c>): every mapping rule, every <c>SchemaError</c> message, and answer projection.
/// </summary>
public class StructuredFixtureTests
{
    private static readonly JsonNode Fixture = TestFiles.Fixture("structured.json");

    public static TheoryData<string> Schemas() => new(Fixture["schemas"]!.AsArray().Select(c => c!["name"]!.GetValue<string>()));

    public static TheoryData<string> Projections() => new(Fixture["projections"]!.AsArray().Select(c => c!["name"]!.GetValue<string>()));

    private static JsonNode Case(string section, string name) =>
        Fixture[section]!.AsArray().Single(c => c!["name"]!.GetValue<string>() == name)!;

    [Theory]
    [MemberData(nameof(Schemas))]
    public void Schema_plans_like_python(string name)
    {
        var c = Case("schemas", name);
        if (name == "oneof_is_python_rejected")
        {
            // The one deliberate difference: Python rejects this property, the .NET extension plans its described consts.
            Assert.StartsWith("properties.a: a free string", c["error"]!.GetValue<string>(), StringComparison.Ordinal);
            Assert.Equal("""{"a": {"type": "choice", "instructions": "What is `a`?", "criteria": {"x": "the x"}}}""",
                PythonJson.Serialize(DecisionSchema.FromJson(c["schema"]).Questions.ToJson()));
            return;
        }
        if (c["error"] is { } error)
        {
            var ex = Assert.Throws<LayaSchemaException>(() => DecisionSchema.FromJson(c["schema"]));
            Assert.Equal(error.GetValue<string>(), ex.Message);
        }
        else
        {
            // Byte-for-byte json.dumps of the questions: key order, labels and instructions all matter to the model.
            Assert.Equal(PythonJson.Serialize(c["questions"]), PythonJson.Serialize(DecisionSchema.FromJson(c["schema"]).Questions.ToJson()));
        }
    }

    [Theory]
    [MemberData(nameof(Projections))]
    public void Answers_project_like_python(string name)
    {
        var c = Case("projections", name);
        var values = DecisionSchema.FromJson(c["schema"]).Project(c["answers"]!.AsObject());
        Assert.Equal(PythonJson.Serialize(c["values"]), PythonJson.Serialize(values));
    }

    [Fact]
    public void Fixture_covers_every_error_message()
    {
        var errors = Fixture["schemas"]!.AsArray().Where(c => c!["error"] is not null).Select(c => c!["error"]!.GetValue<string>()).ToList();
        foreach (var fragment in new[]
                 {
                     "expected a JSON schema object", "the top level must be an object", "'properties' must be a non-empty object",
                     "exceeds MAX_PROPERTIES", "property must be an object", "exceeds MAX_OPTIONS", "'enum' must not be empty",
                     "a free string", "arrays are not supported", "nested objects are not supported", "$ref/recursion",
                     "needs integer 'minimum' and 'maximum'", "is below 'minimum'", "exceeds MAX_SCORE_LEVELS", "unsupported schema",
                 })
            Assert.Contains(errors, e => e.Contains(fragment, StringComparison.Ordinal));
    }
}
