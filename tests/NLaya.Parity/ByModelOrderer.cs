using Xunit.Sdk;
using Xunit.v3;

namespace NLaya.Parity;

/// <summary>
/// Runs a class's test cases grouped by checkpoint, then backend (from the <c>model</c> and <c>backend</c>
/// arguments in the display name), so <see cref="ParityFixture"/> can hold one agent at a time.
/// </summary>
public sealed class ByModelOrderer : ITestCaseOrderer
{
    public IReadOnlyCollection<TTestCase> OrderTestCases<TTestCase>(IReadOnlyCollection<TTestCase> testCases)
        where TTestCase : notnull, ITestCase =>
        testCases
            .OrderBy(t => Argument(t.TestCaseDisplayName, "model"), StringComparer.Ordinal)
            .ThenBy(t => Argument(t.TestCaseDisplayName, "backend"), StringComparer.Ordinal)
            .ThenBy(t => t.TestCaseDisplayName, StringComparer.Ordinal)
            .ToList();

    private static string Argument(string displayName, string name)
    {
        var marker = $"{name}: \"";
        var start = displayName.IndexOf(marker, StringComparison.Ordinal);
        if (start < 0) return "";
        start += marker.Length;
        var end = displayName.IndexOf('"', start);
        return end < 0 ? "" : displayName[start..end];
    }
}
