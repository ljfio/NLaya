using System.Runtime.CompilerServices;

using Microsoft.ML;

namespace NLaya.ML;

/// <summary>
/// ML.NET builds <see cref="SchemaShape"/>s with an internal factory, and <c>SchemaShape.Column</c> has
/// no public constructor, so an estimator outside ML.NET can't describe new columns. This calls
/// that factory, which every ML.NET estimator uses; a rename fails the NLaya.ML tests, not silently.
/// </summary>
internal static class SchemaShapes
{
    [UnsafeAccessor(UnsafeAccessorKind.StaticMethod, Name = "Create")]
    private static extern SchemaShape Create(SchemaShape? _, DataViewSchema schema);

    public static SchemaShape From(DataViewSchema schema) => Create(null, schema);
}
