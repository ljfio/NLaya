"""Generate golden fixtures for NLaya from the reference Python `laya` implementation.

    uv run --python 3.12 --with laya --with tokenizers tools/fixtures/make_fixtures.py

Writes tests/NLaya.Tests/Fixtures/*.json:
  tokenizer_multilingual.json / tokenizer_english.json  text -> ids (HF `tokenizers`, no specials)
  json_states.json                                       state -> json.dumps(ensure_ascii=False)
  sequences.json                                         build_sequence ids + markers
  model_multilingual.json                                batch tensors -> logits / act_logits, predict outputs
"""
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


def main():
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

    # ---- tokenizers
    for name, repo, sub in (("multilingual", REPO, None), ("english", "convaiinnovations/laya", None)):
        path = hf_hub_download(repo, "tokenizer/tokenizer.json")
        tk = Tokenizer.from_file(path)
        dump("tokenizer_%s.json" % name, {
            "repo": repo,
            "cases": [{"text": t, "ids": tk.encode(t, add_special_tokens=False).ids} for t in TEXTS],
        })

    # ---- json states
    dump("json_states.json", {"cases": [{"state": s, "json": serialize_state(s)} for s in STATES]})

    # ---- agent
    agent = laya.load(REPO, device="cpu")
    tok = agent.tok

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

    # ---- raw model outputs for a padded batch of mixed questions
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
            ids, markers = build_sequence(tok, state, q, 1024, 256, truncate_left=isinstance(state, list))
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

    dump("model_multilingual.json", {"repo": REPO, "batches": model_cases, "predicts": predicts,
                                     "predict_batch": batch})


if __name__ == "__main__":
    sys.exit(main())
