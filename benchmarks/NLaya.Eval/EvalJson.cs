using System.Text.Json.Serialization;

using NLaya.Calibration;

namespace NLaya.Eval;

/// <summary>Source-generated JSON for the report, snake_case like laya's result files.</summary>
[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower, NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals)]
[JsonSerializable(typeof(ClassificationReport))]
[JsonSerializable(typeof(IReadOnlyList<double>))]
[JsonSerializable(typeof(List<double>))]
[JsonSerializable(typeof(IReadOnlyDictionary<string, double>))]
internal sealed partial class EvalJson : JsonSerializerContext;
