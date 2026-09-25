using System.Numerics;
using System.Text.Json.Nodes;

namespace NLaya;

/// <summary>
/// One planned schema property (Python's <c>_Field</c>): the question it asks, and how its answer maps
/// back to a value.
/// </summary>
/// <param name="Name">The property name, which is also the question id.</param>
/// <param name="Kind">How the answer is projected.</param>
/// <param name="Question">The question in Python's dict shape.</param>
/// <param name="Options">Choice: each option's label and the schema value it stands for, in order.</param>
/// <param name="Minimum">Score: the value of level 0.</param>
internal sealed record DecisionField(
    string Name,
    QuestionType Kind,
    JsonObject Question,
    IReadOnlyList<(string Label, JsonNode? Value)> Options,
    BigInteger Minimum);
