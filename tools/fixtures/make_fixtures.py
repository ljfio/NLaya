"""Generate golden fixtures for NLaya from the reference Python `laya` implementation.

    git clone https://github.com/NandhaKishorM/laya && git -C laya checkout $(cut -d" " -f1 tools/fixtures/LAYA_COMMIT)
    uv run --python 3.12 --with ./laya --with tokenizers tools/fixtures/make_fixtures.py --laya-tests laya/tests

Writes tests/NLaya.Tests/Fixtures/*.json:
  tokenizer_multilingual.json / tokenizer_english.json  text -> ids (HF `tokenizers`, no specials)
  json_states.json                                       state -> json.dumps(ensure_ascii=False)
  sequences.json                                         build_sequence ids + markers
  model_{multilingual,english,typed_decisions}.json      batch tensors -> logits / act_logits, predict outputs
  lang.json / email.json                                 laya.lang.analyse, Router routing, email cleaning
  router_batch.json                                      Router.route_batch / predict_batch grouping (fake agents)
  src/NLaya/Presets/presets.json                         laya.presets, embedded in the library

Pass --only lang,email,router_batch,presets,tokenizers,sequences,models to regenerate a subset.
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
    if want("router_batch"):
        router_batch_fixture(dump)

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


def router_batch_fixture(dump):
    """Router.route_batch decisions and Router.predict_batch grouping, with fake agents attached (no model)."""
    from laya.router import Router

    q_a = {"intent": {"type": "choice", "instructions": "What does the sender want?",
                      "criteria": {"refund": "money back", "help": "support"}},
           "urgent": {"type": "noul", "instructions": "Is it urgent?"}}
    q_b = {"urgent": {"type": "noul", "instructions": "Is it urgent?"}}
    q_a_reordered = {"urgent": q_a["urgent"], "intent": q_a["intent"]}  # same questions, other order: own group
    requests = [
        {"state": "I was charged twice, please refund me", "questions": q_a},
        {"state": "二重に請求されました", "questions": q_a},
        {"state": "Please help, the app crashes on login", "questions": q_b},
        {"state": "Mein Konto wurde zweimal belastet – bitte erstatten Sie den Betrag.", "questions": q_a},
        {"state": "Refund the duplicate charge", "questions": q_a, "model": "multilingual"},
        {"state": "Hola, necesito ayuda", "questions": q_a, "lang": "es"},
        {"state": "Where is my order?", "questions": q_a_reordered},
        {"state": {"body": "I need a copy of my invoice"}, "questions": q_a, "lang_guess": "fr"},
        {"state": "hello", "questions": q_a, "lang": "de"},
        {"state": "Ich möchte mein Abo kündigen, bitte bestätigen", "questions": q_a},
        {"state": "Where is my order?", "questions": q_a},
    ]
    calls = []

    class FakeAgent:
        def __init__(self, name, lang_temperatures=None):
            self.name = name
            self.lang_temperatures = lang_temperatures or {}

        def predict_batch(self, states, questions, batch_size=None, **kwargs):
            calls.append({"model": self.name, "states": list(states), "questions": list(questions),
                          "lang": kwargs.get("lang"), "batch_size": batch_size})
            return [{"model": "laya-rl-agent", "answers": {}, "usage": {"input_tokens": 1, "output_tokens": 0}}
                    for _ in states]

    router = Router()
    router.attach("english", FakeAgent("english"))
    # Only an agent with language temperatures splits its groups by language.
    router.attach("multilingual", FakeAgent("multilingual", {"de": {"temperature": [1.0, 1.0, 1.0]}}))
    decisions = router.route_batch(requests)
    results = router.predict_batch(requests, batch_size=4)
    dump("router_batch.json", {
        "requests": requests,
        "decisions": [{"model": d["model"], "reason": d["reason"]} for d in decisions],
        "calls": calls,
        "result_models": [r["routing"]["model"] for r in results],
    })


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
