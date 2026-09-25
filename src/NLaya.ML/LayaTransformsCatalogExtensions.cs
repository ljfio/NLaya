using NLaya;
using NLaya.ML;

namespace Microsoft.ML;

/// <summary><c>mlContext.Transforms.Laya(...)</c>: answer Laya questions in an ML.NET pipeline.</summary>
public static class LayaTransformsCatalogExtensions
{
    /// <summary>
    /// Answer <paramref name="questions"/> about each row with <paramref name="predictor"/> (a
    /// <see cref="LayaAgent"/> or a <see cref="NLaya.Routing.Router"/>), reading the state from
    /// <paramref name="inputColumnNames"/>. See <see cref="LayaEstimator"/> for the columns it adds.
    /// </summary>
    public static LayaEstimator Laya(this TransformsCatalog catalog, ILayaPredictor predictor, Questions questions,
        params string[] inputColumnNames) =>
        catalog.Laya(predictor, questions, new LayaTransformerOptions(), inputColumnNames);

    /// <inheritdoc cref="Laya(TransformsCatalog, ILayaPredictor, Questions, string[])"/>
    public static LayaEstimator Laya(this TransformsCatalog catalog, ILayaPredictor predictor, Questions questions,
        LayaTransformerOptions options, params string[] inputColumnNames)
    {
        ArgumentNullException.ThrowIfNull(catalog);
        ArgumentNullException.ThrowIfNull(options);
        return new LayaEstimator(predictor, questions, inputColumnNames, options);
    }
}
