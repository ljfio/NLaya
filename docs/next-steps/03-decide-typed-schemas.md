# 3. `Decide<T>()`: typed values from a C# type

**Goal:** port `laya.structured` (`decide`, `plan_from_json_schema`, `answers_to_json`,
`DecisionResult`) so a caller describes the answer as a C# type and gets a filled-in instance back:

```csharp
public enum Department { Billing, Support, Sales }
public sealed record TicketTriage(
    [property: Description("Which team should handle `body`?")] Department Department,
    [property: Range(0, 2)] int Urgency,
    bool NeedsHuman);

TicketTriage t = agent.Decide<TicketTriage>(email);          // also on Router
DecisionResult<TicketTriage> d = agent.DecideWithDetails<TicketTriage>(email);  // + confidence, probabilities, usage, routing
```

This is the most "native .NET" part of the API: no stringly-typed question ids.

## Use Microsoft libraries

- **Schema generation:** `System.Text.Json.Schema.JsonSchemaExporter.GetJsonSchemaAsNode(options, typeof(T))`.
  It's built into .NET 9+; on `net8.0` it needs the `System.Text.Json` 9.x package, or do
  [step 4a](04-reduce-custom-code.md) first and target only .NET 10.
- **Descriptions and ranges:** use `JsonSchemaExporterOptions.TransformSchemaNode` to copy
  `[Description]` into `description`, and `[Range(min, max)]` into `minimum`/`maximum`
  (`System.ComponentModel.DataAnnotations`).
- **Enums:** serialize as strings (`JsonStringEnumConverter`) so the schema has `enum: ["Billing", ...]`.
  Honour `[JsonStringEnumMemberName]` for custom labels (.NET 9+).
- **Values:** after projection, deserialize the values object with the same
  `JsonSerializerOptions` to get `T`.

Also accept a raw `JsonNode` schema, mirroring Python's `decide(runner, state, schema=dict)`:
`agent.Decide(state, JsonNode schema)` returning a `JsonObject`.

## Port exactly (from `laya/structured.py` at the pinned commit)

- **Limits:** `MAX_PROPERTIES = 32`, `MAX_OPTIONS = 32`, `MAX_SCORE_LEVELS = 10`.
- **Top level:** must be `type: object` (or no type) with a non-empty `properties`.
- **Per property, in this order:**
  - `const` → one-option enum.
  - `enum` → `choice`: labels are `str(value)` (`"null"` for null), criteria has no descriptions,
    and the instructions default to ``What is `name`?``. An enum of only booleans → `noul`.
  - `type` may be a list (nullable): take the first non-`"null"` type.
  - `boolean` → `noul`; the instructions default to ``Is `name` true?``.
  - `integer` / `number` → `score`. This needs integer `minimum` and `maximum`, with
    `span = max - min + 1 <= 10`. The criteria are `["min", ..., "max"]` as strings, and the
    instructions default to ``Score `name` from min to max``.
  - `string` (free), `array`, `object` and `$ref` are rejected. Each raises `SchemaError` with the
    path `properties.<name>` and the Python wording (copy the messages).
- **Projection (`_project`):**
  - `noul` → `Noul >= 0.5`.
  - `score` → `minimum` + the argmax of the probabilities. This is the most likely level, **not**
    the rounded expected score, unless there are no probabilities.
  - `choice` → the original enum value whose label matches.
- **Details (`_details`):**
  - `confidence` is each answer's `confidence`.
  - `probabilities` are `{"false": 1-p, "true": p}` for noul (rounded to 4 places), otherwise the
    answer's probabilities.
  - Also carry `answers`, `usage` and `routing`.

## Tests (golden fixtures, like the rest of the port)

- Extend `tools/fixtures/make_fixtures.py` with a `structured` section. Put a list of JSON schemas
  (valid ones, plus each error case) through `plan_from_json_schema` / `questions_from_json_schema`,
  and record the questions or the `SchemaError` message. `answers_to_json` needs only answers, not
  the model: record projections for hand-written answer dicts. None of this needs the model.
- In C#: schema → questions must match Python's JSON exactly (`Questions.ToJson()`), error messages
  must match, and projections must match. Add `JsonSchemaExporter` tests for the attribute mapping.
- One parity test: `Decide<T>` on a real checkpoint equals Python `decide` for the same schema and
  state (add those to `model_*.json`).

## Done when

`Decide<T>`, `DecideWithDetails<T>` and `Decide(state, JsonNode)` exist on `LayaAgent` and `Router`
(or on `ILayaPredictor` as extension methods, which is thinner). The README gets a "Typed decisions"
section, and the fixtures cover every mapping rule and error.
