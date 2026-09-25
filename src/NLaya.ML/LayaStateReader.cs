using System.Globalization;
using System.Text.Json.Nodes;

using Microsoft.ML;
using Microsoft.ML.Data;

namespace NLaya.ML;

/// <summary>
/// Turns a row's input columns into a <see cref="LayaState"/>: one text column is the state's text,
/// and several columns become a JSON object keyed by column name (like a Python dict state).
/// </summary>
internal sealed class LayaStateReader
{
    private readonly (string Name, Func<JsonNode?> Read)[] _columns;
    private readonly bool _single;

    public LayaStateReader(DataViewRow row, IReadOnlyList<DataViewSchema.Column> columns)
    {
        _columns = columns.Select(c => (c.Name, Reader(row, c))).ToArray();
        _single = columns.Count == 1;
    }

    public LayaState Read()
    {
        if (_single)
        {
            var value = _columns[0].Read();
            return value is JsonValue v && v.TryGetValue<string>(out var text) ? LayaState.FromText(text) : LayaState.FromJson(value);
        }
        var obj = new JsonObject();
        foreach (var (name, read) in _columns) obj[name] = read();
        return LayaState.FromJson(obj);
    }

    /// <summary>The input columns, checked: they must exist and hold text, numbers or booleans.</summary>
    public static DataViewSchema.Column[] Resolve(DataViewSchema schema, IReadOnlyList<string> names)
    {
        if (names.Count == 0) throw new ArgumentException("name at least one input column", nameof(names));
        return names.Select(n =>
        {
            var c = schema.GetColumnOrNull(n) ?? throw new ArgumentOutOfRangeException(nameof(names), $"input column '{n}' not found");
            if (!IsSupported(c.Type))
                throw new ArgumentOutOfRangeException(nameof(names), $"input column '{n}' is {c.Type}; Laya reads text, number and boolean columns");
            return c;
        }).ToArray();
    }

    internal static bool IsSupported(DataViewType type) =>
        type is TextDataViewType or BooleanDataViewType ||
        (type is NumberDataViewType && Type.GetTypeCode(type.RawType) is >= TypeCode.SByte and <= TypeCode.Double);

    private static Func<JsonNode?> Reader(DataViewRow row, DataViewSchema.Column c) => c.Type switch
    {
        TextDataViewType => Getter<ReadOnlyMemory<char>>(row, c, v => JsonValue.Create(v.ToString())),
        BooleanDataViewType => Getter<bool>(row, c, v => JsonValue.Create(v)),
        _ => Type.GetTypeCode(c.Type.RawType) switch
        {
            // Via the float's shortest string, so 0.1f is written as 0.1, not 0.10000000149011612.
            TypeCode.Single => Getter<float>(row, c, v => float.IsFinite(v) ? JsonValue.Create(double.Parse(v.ToString(CultureInfo.InvariantCulture), CultureInfo.InvariantCulture)) : null),
            TypeCode.Double => Getter<double>(row, c, v => double.IsFinite(v) ? JsonValue.Create(v) : null),
            TypeCode.SByte => Getter<sbyte>(row, c, v => JsonValue.Create(v)),
            TypeCode.Byte => Getter<byte>(row, c, v => JsonValue.Create(v)),
            TypeCode.Int16 => Getter<short>(row, c, v => JsonValue.Create(v)),
            TypeCode.UInt16 => Getter<ushort>(row, c, v => JsonValue.Create(v)),
            TypeCode.Int32 => Getter<int>(row, c, v => JsonValue.Create(v)),
            TypeCode.UInt32 => Getter<uint>(row, c, v => JsonValue.Create(v)),
            TypeCode.Int64 => Getter<long>(row, c, v => JsonValue.Create(v)),
            TypeCode.UInt64 => Getter<ulong>(row, c, v => JsonValue.Create(v)),
            _ => throw new ArgumentOutOfRangeException(nameof(c), string.Create(CultureInfo.InvariantCulture, $"column '{c.Name}' is {c.Type}")),
        },
    };

    private static Func<JsonNode?> Getter<T>(DataViewRow row, DataViewSchema.Column c, Func<T, JsonNode?> convert)
    {
        var get = row.GetGetter<T>(c);
        T value = default!;
        return () =>
        {
            get(ref value);
            return convert(value);
        };
    }
}
