"""Generate golden fixtures for NLaya from the reference Python `laya` implementation.

    git clone https://github.com/NandhaKishorM/laya && git -C laya checkout $(cut -d" " -f1 tools/fixtures/LAYA_COMMIT)
    uv run --python 3.12 --with ./laya --with tokenizers tools/fixtures/make_fixtures.py --laya-tests laya/tests

Writes tests/NLaya.Tests/Fixtures/*.json:
  tokenizer_multilingual.json / tokenizer_english.json  text -> ids (HF `tokenizers`, no specials)
  json_states.json                                       state -> json.dumps(ensure_ascii=False)
  sequences.json                                         build_sequence ids + markers
  model_{multilingual,english,typed_decisions}.json      batch tensors -> logits / act_logits, predict outputs
  lang.json / email.json                                 laya.lang.analyse, Router routing, email cleaning
  structured.json                                        laya.structured: schema -> questions / SchemaError, projections
  decide.json                                            Agent.decide(..., return_details=True) on each checkpoint
  src/NLaya/Presets/presets.json                         laya.presets, embedded in the library

Pass --only lang,email,presets,tokenizers,sequences,models,structured,decide to regenerate a subset.
"""
import ast
import glob
import json
import os
import sys

os.environ.setdefault("USE_TF", "0")
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "NLaya.Tests", "Fixtures")
REPO = "convaiinnovations/laya-multilingual"

TEXTS = [
    "",
    "hello",
    "Hello world",
    "I was charged twice for invoice 4411. Please refund me today.",
    "मुझसे इनवॉइस 4411 के लिए दो बार शुल्क लिया गया। कृपया आज ही धनवापसी करें।",
    "二重に請求されました",
    "Mein Konto wurde zweimal belastet – bitte erstatten Sie den Betrag.",
    "لقد تم خصم المبلغ مرتين",
    "  leading and  double   spaces  ",
    "line one\nline two\n\nline four\n",
    "tabs\tand\r\ncrlf",
    "emoji 🙂🚀 and 👩‍👩‍👧",
    "cuneiform 𒀀𒀁 and rare 𐍈 glyphs",
    "literal <eos> and <bos> tokens and <mask> here",
    "{\"body\": \"I was charged twice\", \"id\": 42}",
    "Ünïcödé naïve café résumé ñ ß ø å",
    "a" * 300,
    "URL https://example.com/path?q=1&x=2 and email foo.bar@example.org",
    "1234567890 3.14159 -42 1e10",
    "Ελληνικά, Русский, 한국어, ภาษาไทย, ქართული, Հայերեն",
    " ",
    "\n",
    "▁already▁metaspace",
]

STATES = [
    "plain text",
    {"body": "I was charged twice", "id": 42, "ok": True, "none": None, "pi": 3.14, "big": 1e20,
     "small": 1e-7, "neg": -0.5, "int_float": 2.0},
    {"nested": {"list": [1, 2, {"x": "y"}], "unicode": "日本語 é", "esc": "quote\" backslash\\ nl\n tab\t <>&'"}},
    [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
    ["turn one", "turn two"],
    {"ctrl": "\u0001\u001f\u007f", "emoji": "🙂", "empty": {}, "empty_list": []},
]

QUESTIONS = {
    "department": {"type": "choice", "instructions": "Which team should handle `body`?",
                   "criteria": {"billing": "invoices, payments, refunds",
                                "technical": "bugs and outages", "sales": "pricing"}},
    "refund_requested": {"type": "noul", "instructions": "Does the sender ask for money back?"},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["not urgent", "somewhat urgent", "very urgent"]},
    "labels_list": {"type": "choice", "instructions": "Pick a sentiment",
                    "criteria": ["positive", "negative", "neutral"]},
    "noul_custom": {"type": "noul", "instructions": "Is the customer angry?",
                    "criteria": {"true": "the customer is upset", "false": "calm"},
                    "labels": {"false": "B", "true": "A"}},
    "single": {"type": "choice", "instructions": "Only one option", "criteria": {"only": "the only one"}},
    "dict_ins": {"type": "noul", "instructions": {"ask": "is it German?", "lang": "de"}},
    "rubric": {"type": "choice", "instructions": "Rubric criteria",
               "criteria": {"a": {"desc": "first", "weight": 2}, "b": ["x", "y"], "c": 0, "d": ""}},
}

LONG_STATE = {"body": " ".join(["The customer reports that the invoice was charged twice and asks for a refund."] * 40)}


ONLY = None


def want(part):
    return ONLY is None or part in ONLY


def main():
    global ONLY
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--laya-tests", default=None, help="path to the laya repo's tests/ (for lang/email corpora)")
    args = ap.parse_args()
    ONLY = set(args.only.split(",")) if args.only else None
    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    import laya
    from laya.common import build_sequence, collate_items, serialize_state, QTYPES
    from laya.agent import Agent

    os.makedirs(OUT, exist_ok=True)

    def dump(name, obj):
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1)
        print("wrote", name)

    tests_dir = args.laya_tests
    if tests_dir is None and (want("lang") or want("email")):
        raise SystemExit("--laya-tests <laya checkout>/tests is needed for the lang/email corpora")
    if want("presets"):
        presets_fixture()
    if want("lang"):
        lang_fixture(dump, tests_dir)
    if want("email"):
        email_fixture(dump, tests_dir)

    # ---- tokenizers
    for name, repo, sub in (("multilingual", REPO, None), ("english", "convaiinnovations/laya", None)):
        if not want("tokenizers"):
            break
        path = hf_hub_download(repo, "tokenizer/tokenizer.json")
        tk = Tokenizer.from_file(path)
        dump("tokenizer_%s.json" % name, {
            "repo": repo,
            "cases": [{"text": t, "ids": tk.encode(t, add_special_tokens=False).ids} for t in TEXTS],
        })

    # ---- json states
    if want("sequences"):
        dump("json_states.json", {"cases": [{"state": s, "json": serialize_state(s)} for s in STATES]})

    if want("sequences"):
        sequences_fixture(dump, build_sequence, Agent)



    for name, repo, sub in MODELS:
        if want("models"):
            model_fixture(name, repo, sub, dump)

    if want("structured"):
        structured_fixture(dump)
    if want("decide"):
        decide_fixture(dump)


def sequences_fixture(dump, build_sequence, Agent):
    import laya
    tok = laya.load(REPO, device="cpu").tok
    seqs = []
    for qid, qdef in QUESTIONS.items():
        q = Agent._to_internal(qdef)
        for state, max_len, head_max_len in ((STATES[1], 1024, 256), (STATES[3], 1024, 256),
                                             (LONG_STATE, 64, 256), (STATES[4], 40, 32),
                                             ("short", 1024, 16), (LONG_STATE, 1024, 256)):
            ids, markers = build_sequence(tok, state, q, max_len, head_max_len,
                                          truncate_left=isinstance(state, list))
            seqs.append({"qid": qid, "question": qdef, "state": state, "max_len": max_len,
                         "head_max_len": head_max_len, "ids": ids, "markers": markers})
    dump("sequences.json", {"cases": seqs})


MODELS = [
    ("multilingual", REPO, None),
    ("english", "convaiinnovations/laya", None),
    ("typed_decisions", "convaiinnovations/laya", "typed-decisions"),
]


def model_fixture(name, repo, sub, dump):
    import torch
    import laya
    from laya.common import build_sequence, collate_items, QTYPES
    from laya.agent import Agent

    agent = laya.load(repo, device="cpu", subfolder=sub)
    tok = agent.tok
    model_cases = []
    batch_specs = [
        ({"body": "I was charged twice for invoice 4411. Please refund me today."}, ["department", "refund_requested", "urgency"]),
        (STATES[1], ["labels_list", "single", "noul_custom"]),
        (LONG_STATE, ["department", "urgency"]),
    ]
    for state, qids in batch_specs:
        items = []
        for qid in qids:
            q = Agent._to_internal(QUESTIONS[qid])
            ids, markers = build_sequence(tok, state, q, agent.cfg.get("max_len", 512), agent.cfg.get("head_max_len", 192),
                                          truncate_left=isinstance(state, list))
            items.append({"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]]})
        b = collate_items([items], tok.pad_token_id)
        with torch.no_grad():
            hidden = agent.model.encoder(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state
            logits, act = agent.model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
        model_cases.append({
            "input_ids": b["input_ids"].tolist(),
            "attention_mask": b["attention_mask"].tolist(),
            "marker_pos": b["marker_pos"].tolist(),
            "marker_mask": b["marker_mask"].tolist(),
            "qtype": b["qtype"].tolist(),
            "hidden_head": hidden[:, :4, :8].tolist(),
            "hidden_mean_abs": float(hidden.abs().mean()),
            "logits": logits.tolist(),
            "act_logits": act.tolist(),
        })

    predicts = []
    for state in ({"body": "मुझसे इनवॉइस 4411 के लिए दो बार शुल्क लिया गया। कृपया आज ही धनवापसी करें।"},
                  "二重に請求されました", STATES[3], LONG_STATE):
        predicts.append({"state": state, "questions": QUESTIONS, "result": agent.predict(state, QUESTIONS)})
    batch_states = ["I was charged twice", {"body": "Die App stürzt ständig ab"}, LONG_STATE]
    bq = {k: QUESTIONS[k] for k in ("department", "refund_requested", "urgency")}
    batch = {"states": batch_states, "questions": bq, "results": agent.predict_batch(batch_states, bq)}
    dump("model_%s.json" % name, {"repo": repo, "subfolder": sub, "batches": model_cases, "predicts": predicts,
                                  "predict_batch": batch})
    del agent


# ---- laya.structured: every mapping rule and every SchemaError message, no model needed

STRUCTURED_SCHEMA = {
    "type": "object",
    "properties": {
        "department": {"type": "string", "enum": ["billing", "support", "sales"], "description": "Which team?"},
        "urgency": {"type": "integer", "minimum": 0, "maximum": 2},
        "needs_human": {"type": "boolean"},
        "priority": {"enum": [1, 2, 3]},
    },
}

MIXED_ENUM = [1, "a", None, True, 2.5, 1e20, -0.0, [1, "x"], {"k": [None, False]}, "it's"]


def _props(**props):
    return {"type": "object", "properties": props}


STRUCTURED_SCHEMAS = [
    ("laya_test_schema", STRUCTURED_SCHEMA),
    ("no_root_type", {"properties": {"ok": {"type": "boolean"}}}),
    ("const_string", _props(a={"const": "only"})),
    ("const_null", _props(a={"const": None})),
    ("const_bool", _props(a={"const": True, "description": "Is it on?"})),
    ("const_number", _props(a={"const": 3})),
    ("enum_mixed", _props(a={"enum": MIXED_ENUM})),
    ("enum_duplicate_labels", _props(a={"enum": [1, "1", 2]})),
    ("enum_all_bool", _props(a={"enum": [True, False]})),
    ("enum_wins_over_type", _props(a={"type": "string", "enum": ["x", "y"], "minimum": 0})),
    ("const_wins_over_enum", _props(a={"const": "c", "enum": ["x", "y"]})),
    ("enum_32_options", _props(a={"enum": ["v%d" % i for i in range(32)]})),
    ("nullable_integer", _props(a={"type": ["integer", "null"], "minimum": 1, "maximum": 3})),
    ("nullable_null_first", _props(a={"type": ["null", "boolean"]})),
    ("number_score", _props(a={"type": "number", "minimum": -2, "maximum": 2})),
    ("score_one_level", _props(a={"type": "integer", "minimum": 5, "maximum": 5})),
    ("score_ten_levels", _props(a={"type": "integer", "minimum": 1, "maximum": 10})),
    ("score_bool_bounds", _props(a={"type": "integer", "minimum": True, "maximum": 3})),
    ("description_empty", _props(a={"type": "boolean", "description": ""})),
    ("description_zero", _props(a={"type": "boolean", "description": 0})),
    ("description_object", _props(a={"type": "boolean", "description": {"ask": "is it on?", "lang": "en"}})),
    ("description_on_score", _props(a={"type": "integer", "minimum": 0, "maximum": 2, "description": "How bad?"})),
    ("unicode_names", _props(**{"urgência": {"type": "boolean"}, "名前": {"enum": ["é", "日本"]}})),
    ("properties_32", _props(**{("p%d" % i): {"type": "boolean"} for i in range(32)})),
    ("oneof_is_python_rejected", _props(a={"type": "string", "oneOf": [{"const": "x", "description": "the x"}]})),
    # errors
    ("err_not_object_list", [1, 2]),
    ("err_not_object_str", "schema"),
    ("err_not_object_none", None),
    ("err_root_array", {"type": "array"}),
    ("err_root_nullable_object", {"type": ["object", "null"], "properties": {"a": {"type": "boolean"}}}),
    ("err_no_properties", {"type": "object"}),
    ("err_empty_properties", {"type": "object", "properties": {}}),
    ("err_properties_list", {"type": "object", "properties": [{"type": "boolean"}]}),
    ("err_properties_33", _props(**{("p%d" % i): {"type": "boolean"} for i in range(33)})),
    ("err_prop_str", _props(a="boolean")),
    ("err_prop_int", _props(a=1)),
    ("err_prop_float", _props(a=1.5)),
    ("err_prop_bool", _props(a=True)),
    ("err_prop_none", _props(a=None)),
    ("err_prop_list", _props(a=[{"type": "boolean"}])),
    ("err_enum_33", _props(a={"enum": ["v%d" % i for i in range(33)]})),
    ("err_enum_empty", _props(a={"enum": []})),
    ("err_free_string", _props(a={"type": "string"})),
    ("err_nullable_string", _props(a={"type": ["string", "null"]})),
    ("err_array", _props(a={"type": "array", "items": {"type": "string"}})),
    ("err_object", _props(a={"type": "object", "properties": {}})),
    ("err_ref", _props(a={"$ref": "#/$defs/X"})),
    ("err_object_with_ref", _props(a={"type": "object", "$ref": "#/$defs/X"})),
    ("err_unbounded", _props(a={"type": "integer", "minimum": 0})),
    ("err_float_bounds", _props(a={"type": "integer", "minimum": 0.0, "maximum": 2.0})),
    ("err_max_below_min", _props(a={"type": "integer", "minimum": 3, "maximum": 1})),
    ("err_too_wide", _props(a={"type": "integer", "minimum": 0, "maximum": 100})),
    ("err_too_wide_11", _props(a={"type": "number", "minimum": -5, "maximum": 5})),
    ("err_unsupported_null", _props(a={"type": "null"})),
    ("err_unsupported_empty", _props(a={})),
    ("err_unsupported_only_null", _props(a={"type": ["null"]})),
    ("err_unsupported_repr", _props(a={"type": "weird", "x": [1, 2.5, None, True, "q'uote", 'd"q', "é\n\t\\"]})),
    ("err_second_property", _props(ok={"type": "boolean"}, bad={"type": "string"})),
]

STRUCTURED_PROJECTIONS = [
    ("laya_test_answers", STRUCTURED_SCHEMA, {
        "department": {"type": "choice", "choice": "billing", "confidence": 0.9,
                       "probabilities": {"billing": 0.9, "support": 0.1, "sales": 0.0}},
        "urgency": {"type": "score", "score": 1.2, "confidence": 0.5,
                    "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}, "legend": {}},
        "needs_human": {"type": "noul", "noul": 0.8, "confidence": 0.8},
        "priority": {"type": "choice", "choice": "2", "confidence": 0.7,
                     "probabilities": {"1": 0.2, "2": 0.7, "3": 0.1}},
    }),
    ("noul_threshold", _props(lo={"type": "boolean"}, mid={"type": "boolean"}, hi={"type": "boolean"}, none={"type": "boolean"}), {
        "lo": {"type": "noul", "noul": 0.4999}, "mid": {"type": "noul", "noul": 0.5},
        "hi": {"type": "noul", "noul": 0.9}, "none": {"type": "noul"},
    }),
    ("score_argmax_ties_first", _props(s={"type": "integer", "minimum": 1, "maximum": 3}), {
        "s": {"type": "score", "score": 2.9, "probabilities": {"0": 0.4, "1": 0.2, "2": 0.4}},
    }),
    ("score_missing_level_key", _props(s={"type": "integer", "minimum": -1, "maximum": 1}), {
        "s": {"type": "score", "score": 0.0, "probabilities": {"1": 0.3, "2": 0.1, "x": 0.9}},
    }),
    ("score_rounds_half_even", _props(a={"type": "integer", "minimum": 1, "maximum": 4},
                                      b={"type": "integer", "minimum": 1, "maximum": 4},
                                      c={"type": "integer", "minimum": 0, "maximum": 2},
                                      d={"type": "integer", "minimum": 0, "maximum": 2}), {
        "a": {"type": "score", "score": 1.5}, "b": {"type": "score", "score": 2.5, "probabilities": {}},
        "c": {"type": "score", "score": 0.49}, "d": {"type": "score"},
    }),
    ("choice_values_keep_type", _props(m={"enum": MIXED_ENUM}, n={"enum": MIXED_ENUM}, o={"enum": MIXED_ENUM},
                                       p={"enum": MIXED_ENUM}, q={"enum": MIXED_ENUM}, r={"enum": MIXED_ENUM}), {
        "m": {"type": "choice", "choice": "None"}, "n": {"type": "choice", "choice": "null"},
        "o": {"type": "choice", "choice": "True"}, "p": {"type": "choice", "choice": "1e+20"},
        "q": {"type": "choice", "choice": "[1, 'x']"}, "r": {"type": "choice", "choice": "{'k': [None, False]}"},
    }),
    ("choice_duplicate_label_first_wins", _props(a={"enum": [1, "1", 2]}), {"a": {"type": "choice", "choice": "1"}}),
    ("choice_unknown_label", _props(a={"enum": ["x", "y"]}, b={"enum": ["x", "y"]}), {
        "a": {"type": "choice", "choice": "z"}, "b": {"type": "choice"},
    }),
    ("choice_numeric_choice", _props(a={"enum": [1, 2]}), {"a": {"type": "choice", "choice": 2}}),
    ("missing_and_extra_answers", _props(a={"type": "boolean"}, b={"type": "boolean"}), {
        "b": {"type": "noul", "noul": 0.7}, "extra": {"type": "noul", "noul": 0.9},
    }),
    ("const_bool_is_noul", _props(a={"const": False}), {"a": {"type": "noul", "noul": 0.6}}),
]


def structured_fixture(dump):
    from laya.structured import SchemaError, answers_to_json, questions_from_json_schema
    schemas = []
    for name, schema in STRUCTURED_SCHEMAS:
        case = {"name": name, "schema": schema}
        try:
            case["questions"] = questions_from_json_schema(schema)
        except SchemaError as e:
            case["error"] = str(e)
        schemas.append(case)
    projections = [{"name": name, "schema": schema, "answers": answers, "values": answers_to_json(answers, schema)}
                   for name, schema, answers in STRUCTURED_PROJECTIONS]
    dump("structured.json", {"schemas": schemas, "projections": projections})


DECIDE_SCHEMA = {
    "type": "object",
    "properties": {
        "department": {"enum": ["billing", "support", "sales"], "description": "Which team should handle `body`?"},
        "urgency": {"type": "integer", "minimum": 1, "maximum": 3, "description": "How urgent is this?"},
        "refund_requested": {"type": "boolean", "description": "Does the sender ask for money back?"},
        "priority": {"enum": [1, 2, 3]},
        "needs_human": {"type": "boolean"},
    },
}

DECIDE_STATES = [
    {"body": "I was charged twice for invoice 4411. Please refund me today."},
    "The app crashes every time I open the settings page, can someone look at it?",
    {"body": "Mein Konto wurde zweimal belastet – bitte erstatten Sie den Betrag."},
]


def decide_fixture(dump):
    import dataclasses
    import laya
    cases = []
    for name, repo, sub in MODELS:
        agent = laya.load(repo, device="cpu", subfolder=sub)
        for state in DECIDE_STATES:
            details = agent.decide(state, DECIDE_SCHEMA, return_details=True)
            cases.append({"model": name, "state": state, "details": dataclasses.asdict(details)})
        del agent
    dump("decide.json", {"schema": DECIDE_SCHEMA, "cases": cases})


def harvest_strings(pattern, max_len=3000):
    """Every string literal in the upstream test files: a ready-made corpus of edge cases."""
    out = []
    for path in sorted(glob.glob(pattern)):
        for node in ast.walk(ast.parse(open(path, encoding="utf-8").read())):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and 0 < len(node.value) <= max_len:
                out.append(node.value)
    return list(dict.fromkeys(out))


def lang_fixture(dump, tests_dir):
    from laya.lang import analyse
    from laya.router import Router, match_typed_decisions_workflow

    texts = harvest_strings(os.path.join(tests_dir, "test_lang*.py")) + harvest_strings(os.path.join(tests_dir, "test_router*.py")) \
        + harvest_strings(os.path.join(tests_dir, "test_blank_lang_routing.py")) + harvest_strings(os.path.join(tests_dir, "test_identifier_complexity.py")) \
        + TEXTS + [s for s in STATES]
    states = texts + [
        {"subject": "Invoice", "body": "Olá, fui cobrado duas vezes pela fatura 4411.\nTraceback (most recent call last): error in module"},
        {"ticket": {"title": "Login broken", "description": "Não consigo entrar na minha conta desde ontem, já tentei tudo"}},
        ["Hello there, I need help with my order", "Bonjour, je voudrais annuler ma commande s'il vous plaît"],
        {"a": 1, "b": None, "c": [True, 2.5]},
    ]
    router = Router()
    cases = []
    for st in states:
        det = analyse(st)
        d = router._route(st, None)
        cases.append({"state": st, "analyse": det, "route": {"model": d["model"], "reason": d["reason"], "repo": d["repo"]}})
    explicit = []
    for kw in ({"model": "en"}, {"model": "ml"}, {"task": "typed_decisions"}, {"lang": "de-AT"}, {"lang": "en_US.UTF-8"},
               {"lang": "C.UTF-8"}, {"lang": " "}, {"lang_guess": "fr"}, {"lang_guess": "und"}):
        d = router._route("Some English text for the router to consider", None, **kw)
        explicit.append({"kwargs": kw, "model": d["model"], "reason": d["reason"]})
    wf = {"customer_service": {"action": 1, "category": 1, "churn_risk": 1, "needs_human": 1, "urgency": 1}}
    auto = Router(auto_task_detection=True)._route("hello there my friend", wf["customer_service"])
    dump("lang.json", {"cases": cases, "explicit": explicit,
                       "workflow": {"questions": list(wf["customer_service"]), "model": auto["model"], "reason": auto["reason"],
                                    "workflow": match_typed_decisions_workflow(wf["customer_service"])}})


def email_fixture(dump, tests_dir):
    from laya.email import clean_email_body, email_state
    bodies = harvest_strings(os.path.join(tests_dir, "test_email.py"), max_len=20000)
    bodies += ["Hi team,\n\nPlease refund invoice 4411.\n\nThanks,\nAna\n\nSent from my iPhone",
               "Olá,\n\nPreciso de ajuda.\n\nEm seg., 3 de jun. de 2024 às 10:00, Suporte <s@x.com> escreveu:\n> antiga mensagem",
               "x" * 5000]
    cases = [{"body": b, "clean": clean_email_body(b), "state": email_state("Subject line", b, sender="a@b.com")} for b in bodies]
    dump("email.json", {"cases": cases,
                        "raw_state": email_state("S", "  body  ", clean=False, priority="high")})


def data_resources():
    """The tables and patterns laya.lang / laya.email are driven by, embedded in NLaya as JSON."""
    import re
    from laya import lang, email
    root = os.path.join(os.path.dirname(__file__), "..", "..", "src", "NLaya")

    def write(rel, obj):
        with open(os.path.join(root, rel), "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1)
        print("wrote", rel)

    write("Lang/lang_data.json", {
        "script_ranges": [[name, [list(r) for r in ranges]] for name, ranges in lang._SCRIPT_RANGES],
        "stopwords": [[lg, sorted(words)] for lg, words in lang._STOP.items()],
        "non_en_diacritics": "".join(sorted(lang._NON_EN_DIACRITICS)),
        "non_en_diacritic_rate": lang.NON_EN_DIACRITIC_RATE,
        "non_latin_fraction": lang.NON_LATIN_FRACTION,
        "non_latin_min_fraction": lang.NON_LATIN_MIN_FRACTION,
        "non_latin_min_letters": lang.NON_LATIN_MIN_LETTERS,
    })

    def pat(r):
        return {"pattern": r.pattern, "ignore_case": bool(r.flags & re.I)}

    write("Email/email_data.json", {
        "quote_headers": [pat(r) for r in email._QUOTE_HEADERS],
        "attribution_tail": pat(email._ATTRIBUTION_TAIL),
        "attribution_head": pat(email._ATTRIBUTION_HEAD),
        "header_from_name": pat(email._HEADER_FROM_NAME),
        "header_next": pat(email._HEADER_NEXT),
        "signature_markers": [pat(r) for r in email._SIGNATURE_MARKERS],
        "device_footer": pat(email._DEVICE_FOOTER),
        "disclaimer": pat(email._DISCLAIMER),
    })


def presets_fixture():
    data_resources()
    from laya import presets
    out = {name: getattr(presets, name)() for name in
           ("triage_questions", "email_questions", "guard_questions", "moderation_questions", "router_questions")}
    path = os.path.join(os.path.dirname(__file__), "..", "..", "src", "NLaya", "Presets", "presets.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("wrote presets.json")


if __name__ == "__main__":
    sys.exit(main())
