namespace NLaya;

/// <summary>
/// A schema can't be expressed as Laya questions. Port of Python's <c>laya.structured.SchemaError</c>:
/// the message is Python's, and names the offending path (for example <c>properties.name: ...</c>).
/// </summary>
public sealed class LayaSchemaException(string message) : ArgumentException(message);
