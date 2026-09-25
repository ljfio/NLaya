using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Diagnostics.CodeAnalysis;
using System.Reflection;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Schema;
using System.Text.Json.Serialization.Metadata;

namespace NLaya;

/// <summary>
/// A C# type's JSON schema, ready for <see cref="DecisionPlanner"/>: <see cref="JsonSchemaExporter"/>
/// plus the attributes it doesn't map. <c>[Description]</c> becomes <c>description</c>, <c>[Range]</c>
/// becomes <c>minimum</c>/<c>maximum</c>, and <c>[Description]</c> on enum members becomes a
/// <c>oneOf</c> of described consts. AOT-safe given a source-generated <see cref="JsonTypeInfo"/>.
/// </summary>
internal static class TypeSchema
{
    private static readonly JsonSchemaExporterOptions ExporterOptions = new()
    {
        TreatNullObliviousAsNonNullable = true,
        TransformSchemaNode = Transform,
    };

    public static JsonObject Export(JsonTypeInfo typeInfo)
    {
        if (typeInfo.GetJsonSchemaAsNode(ExporterOptions) is not JsonObject schema)
            throw new LayaSchemaException($"{typeInfo.Type.Name} has no object schema; decide on a class, record or struct with properties");
        // Reference types export as nullable (["object", "null"]); the value decided is never null.
        if (schema["type"] is JsonArray types)
            schema["type"] = types.FirstOrDefault(t => t?.GetValueKind() != JsonValueKind.String || t.GetValue<string>() != "null")?.DeepClone();
        return schema;
    }

    private static JsonNode Transform(JsonSchemaExporterContext ctx, JsonNode node)
    {
        if (node is not JsonObject o) return node;
        if (ctx.PropertyInfo is { } property)
        {
            foreach (var attribute in Attributes(property))
            {
                switch (attribute)
                {
                    case DescriptionAttribute d when !string.IsNullOrEmpty(d.Description):
                        o["description"] = d.Description;
                        break;
                    case RangeAttribute r:
                        // RangeAttribute's bounds are object: int, double, or text with an operand type.
                        if (Bound(r.Minimum, r.OperandType) is { } min) o["minimum"] = min;
                        if (Bound(r.Maximum, r.OperandType) is { } max) o["maximum"] = max;
                        break;
                }
            }
        }

        var type = Nullable.GetUnderlyingType(ctx.TypeInfo.Type) ?? ctx.TypeInfo.Type;
        if (type.IsEnum)
        {
            if (o["enum"] is not JsonArray values)
                throw new LayaSchemaException(
                    $"{Path(ctx)}: enum {type.Name} serializes as a number; serialize enums as strings " +
                    "(JsonStringEnumConverter, or JsonSourceGenerationOptions.UseStringEnumConverter = true)");
            var described = EnumDescriptions(type, ctx.TypeInfo.Options.GetTypeInfo(type));
            if (described.Count > 0)
            {
                // Replace the plain enum with the standard described form; the planner's oneOf extension reads it.
                var oneOf = new JsonArray();
                foreach (var v in values)
                {
                    var item = new JsonObject { ["const"] = v?.DeepClone() };
                    if (v is JsonValue s && s.GetValueKind() == JsonValueKind.String && described.TryGetValue(s.GetValue<string>(), out var d))
                        item["description"] = d;
                    oneOf.Add((JsonNode)item);
                }
                o.Remove("enum");
                o["oneOf"] = oneOf;
            }
        }
        return o;
    }

    private static IEnumerable<object> Attributes(JsonPropertyInfo property)
    {
        // A positional record's attributes land on the parameter unless written [property: ...].
        var fromProperty = property.AttributeProvider?.GetCustomAttributes(true) ?? [];
        var fromParameter = property.AssociatedParameter?.AttributeProvider?.GetCustomAttributes(true) ?? [];
        return fromParameter.Concat(fromProperty);
    }

    private static JsonNode? Bound(object? value, Type operandType) => value switch
    {
        int i => i,
        long l => l,
        double d => d,
        string s when IsIntegral(operandType) && long.TryParse(s, System.Globalization.CultureInfo.InvariantCulture, out var l) => l,
        string s when double.TryParse(s, System.Globalization.CultureInfo.InvariantCulture, out var d) => d,
        _ => null,
    };

    private static bool IsIntegral(Type t) =>
        t == typeof(int) || t == typeof(long) || t == typeof(short) || t == typeof(byte) ||
        t == typeof(sbyte) || t == typeof(uint) || t == typeof(ulong) || t == typeof(ushort);

    /// <summary>Serialized enum name -> its member's <c>[Description]</c>.</summary>
    [UnconditionalSuppressMessage("Trimming", "IL2070",
        Justification = "Enum fields are never trimmed: they are kept for Enum.ToString and parsing, and the serializer already names them.")]
    private static Dictionary<string, string> EnumDescriptions(Type enumType, JsonTypeInfo enumTypeInfo)
    {
        var map = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var field in enumType.GetFields(BindingFlags.Public | BindingFlags.Static))
        {
            if (field.GetCustomAttribute<DescriptionAttribute>()?.Description is not { Length: > 0 } description) continue;
            // The name the serializer writes, which honours [JsonStringEnumMemberName] and naming policies.
            if (JsonSerializer.SerializeToNode(field.GetValue(null), enumTypeInfo) is JsonValue name &&
                name.GetValueKind() == JsonValueKind.String)
                map[name.GetValue<string>()] = description;
        }
        return map;
    }

    private static string Path(JsonSchemaExporterContext ctx) =>
        ctx.PropertyInfo is { } p ? $"properties.{p.Name}" : ctx.TypeInfo.Type.Name;
}
