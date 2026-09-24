using System.Collections;
using System.Diagnostics.CodeAnalysis;

namespace NLaya;

/// <summary>
/// A string-keyed dictionary that keeps insertion order, like a Python dict. Question and
/// answer order is significant: it decides row order in the batch and key order in results.
/// </summary>
public class OrderedMap<TValue> : IReadOnlyDictionary<string, TValue>
{
    private readonly List<string> _keys = new();
    private readonly Dictionary<string, TValue> _map = new(StringComparer.Ordinal);

    public OrderedMap() { }

    public OrderedMap(IEnumerable<KeyValuePair<string, TValue>> items)
    {
        foreach (var (k, v) in items) this[k] = v;
    }

    public TValue this[string key]
    {
        get => _map[key];
        set
        {
            if (!_map.ContainsKey(key)) _keys.Add(key);
            _map[key] = value;
        }
    }

    public void Add(string key, TValue value)
    {
        if (_map.ContainsKey(key)) throw new ArgumentException($"duplicate key '{key}'", nameof(key));
        this[key] = value;
    }

    public bool Remove(string key)
    {
        if (!_map.Remove(key)) return false;
        _keys.Remove(key);
        return true;
    }

    public IEnumerable<string> Keys => _keys;
    public IEnumerable<TValue> Values => _keys.Select(k => _map[k]);
    public int Count => _keys.Count;
    public bool ContainsKey(string key) => _map.ContainsKey(key);
    public bool TryGetValue(string key, [MaybeNullWhen(false)] out TValue value) => _map.TryGetValue(key, out value);

    public IEnumerator<KeyValuePair<string, TValue>> GetEnumerator()
    {
        foreach (var k in _keys) yield return new(k, _map[k]);
    }

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}
