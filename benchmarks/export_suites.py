"""Export laya's benchmark suites as JSONL for NLaya.Eval.

    uv run --python 3.12 --with datasets benchmarks/export_suites.py --out benchmarks/suites
    uv run --python 3.12 --with datasets benchmarks/export_suites.py --out benchmarks/suites --only typed_decisions,en.ag_news

Each suite is built exactly as laya's benchmark notebook builds it (research/scripts/build_benchmark_nb.py
at the commit in tools/fixtures/LAYA_COMMIT): the same datasets, splits, sample counts, question wording
and seed, so NLaya's numbers can be set next to BENCHMARKS.md. One line per case:

    {"state": ..., "questions": {python dict}, "gold": {"qid": {"index": 2, "soft": [..]}}, "workflow": "..."}

`index` is the correct option's position in the question's options (for a noul, 1 is true); `soft` is
the typed-decisions teacher distribution, when there is one. Datasets are downloaded by `datasets`
into its own cache; nothing here touches NLaya's model cache. `en.banking77_full` uses a dataset
loading script, which `datasets` 4 no longer runs; export it with `--with "datasets<4"`.
"""
import argparse
import json
import os
import random

SEED = 13
N_OPTS = 20       # options per many-label choice question (gold + 19 distractors)
PER_LANG = 300    # cases per language
MASSIVE_LANGS = ["en", "de", "fr", "es", "pt", "ru", "tr", "ar", "hi", "ta", "zh-CN", "ja", "ko", "sw"]
XNLI_LANGS = ["en", "de", "fr", "es", "ru", "tr", "ar", "hi", "ur", "vi", "th", "el", "bg", "zh", "sw"]
NLI_CRIT = {"entailment": "the premise implies the hypothesis is true",
            "neutral": "the premise neither implies nor contradicts the hypothesis",
            "contradiction": "the premise implies the hypothesis is false"}


def build_choice_questions(rng, gold, all_labels, n_opts, instructions, qid="label"):
    pool = [x for x in all_labels if x != gold]
    keys = [gold] + rng.sample(pool, min(n_opts - 1, len(pool)))
    rng.shuffle(keys)
    crit = {k: k.replace("_", " ").replace(".", ": ") for k in keys}
    return {qid: {"type": "choice", "instructions": instructions, "criteria": crit}}, keys.index(gold)


def typed_decisions(load_dataset):
    td = load_dataset("LocalLLaMA/typed-decisions", "all", split="test")
    for r in td:
        questions = json.loads(r["questions"]) if isinstance(r["questions"], str) else r["questions"]
        gold = json.loads(r["gold"]) if isinstance(r["gold"], str) else r["gold"]
        state = r["state"]
        try:
            state = json.loads(state)
        except Exception:
            pass
        gmap = {}
        for qid, qdef in questions.items():
            g = gold[qid]
            if qdef["type"] == "choice":
                keys = list(qdef["criteria"].keys())
                gmap[qid] = {"index": keys.index(str(g["label"])),
                             "soft": [float(g.get("probabilities", {}).get(k, 0.0)) for k in keys]}
            elif qdef["type"] == "noul":
                pt = float(g.get("probabilities", {}).get("true", g.get("noul", 0.5)))
                gmap[qid] = {"index": 1 if str(g["label"]).lower() == "true" else 0, "soft": [1 - pt, pt]}
            else:
                n = len(qdef["criteria"])
                gmap[qid] = {"index": int(g["label"]),
                             "soft": [float(g.get("probabilities", {}).get(str(i), 0.0)) for i in range(n)]}
        yield {"state": state, "questions": questions, "gold": gmap, "workflow": r["workflow"]}


def massive(load_dataset, task, instructions, lang):
    d = load_dataset(task, lang, split="test")
    labels = sorted(set(d["label_text"]))
    rng = random.Random(SEED)
    for r in list(d)[:PER_LANG]:
        qs, gi = build_choice_questions(rng, r["label_text"], labels, N_OPTS, instructions)
        yield {"state": {"utterance": r["text"]}, "questions": qs, "gold": {"label": {"index": gi}}}


def xnli(load_dataset, lang):
    d = load_dataset("facebook/xnli", lang, split="test")
    for r in list(d)[:PER_LANG]:
        qs = {"relation": {"type": "choice", "instructions": "What is the relationship between `premise` and `hypothesis`?",
                           "criteria": dict(NLI_CRIT)}}
        yield {"state": {"premise": r["premise"], "hypothesis": r["hypothesis"]}, "questions": qs,
               "gold": {"relation": {"index": int(r["label"])}}}


def sst5(load_dataset):
    d = load_dataset("SetFit/sst5", split="test")
    crit = ["very negative", "negative", "neutral", "positive", "very positive"]
    for r in list(d)[:600]:
        yield {"state": {"text": r["text"]},
               "questions": {"sentiment": {"type": "score", "instructions": "How positive is the sentiment of `text`?",
                                           "criteria": crit}},
               "gold": {"sentiment": {"index": int(r["label"])}}}


def emotion(load_dataset):
    d = load_dataset("dair-ai/emotion", "split", split="test")
    names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    for r in list(d)[:600]:
        yield {"state": {"text": r["text"]},
               "questions": {"emotion": {"type": "choice", "instructions": "Which emotion is most strongly expressed in `text`?",
                                         "criteria": {n: None for n in names}}},
               "gold": {"emotion": {"index": int(r["label"])}}}


def prompt_injections(load_dataset):
    d = load_dataset("deepset/prompt-injections", split="test")
    for r in list(d):
        yield {"state": {"text": r["text"]},
               "questions": {"injection": {"type": "noul",
                                           "instructions": "Does `text` try to inject or override instructions given to an AI system?"}},
               "gold": {"injection": {"index": int(r["label"])}}}


def banking77(load_dataset):
    # PolyAI/banking77 is a loading-script dataset: it needs `--with "datasets<4"`.
    d = load_dataset("PolyAI/banking77", split="test")
    names = d.features["label"].names
    for r in list(d)[:500]:
        yield {"state": {"message": r["text"]},
               "questions": {"intent": {"type": "choice", "instructions": "Which banking intent does `message` express?",
                                        "criteria": {n.replace("_", " "): None for n in names}}},
               "gold": {"intent": {"index": int(r["label"])}}}


def ag_news(load_dataset):
    d = load_dataset("fancyzhx/ag_news", split="test")
    crit = {"world": "world news and international politics", "sports": "sports",
            "business": "business and economy", "sci_tech": "science and technology"}
    for r in list(d)[:600]:
        yield {"state": {"article": r["text"]},
               "questions": {"topic": {"type": "choice", "instructions": "What is the topic of `article`?", "criteria": dict(crit)}},
               "gold": {"topic": {"index": int(r["label"])}}}


def boolq(load_dataset):
    d = load_dataset("google/boolq", split="validation")
    for r in list(d)[:600]:
        yield {"state": {"passage": r["passage"], "question": r["question"]},
               "questions": {"answer": {"type": "noul", "instructions": "Based on `passage`, is the answer to `question` yes?"}},
               "gold": {"answer": {"index": int(bool(r["answer"]))}}}


def suites(load_dataset):
    yield "typed_decisions", lambda: typed_decisions(load_dataset)
    for name, fn in (("en.sst5", sst5), ("en.emotion", emotion), ("en.prompt_injections", prompt_injections),
                     ("en.banking77_full", banking77), ("en.ag_news", ag_news), ("en.boolq", boolq)):
        yield name, (lambda f=fn: f(load_dataset))
    for task, short, instr in (("mteb/amazon_massive_intent", "massive_intent", "What is the user asking for in `utterance`?"),
                               ("mteb/amazon_massive_scenario", "massive_scenario", "Which domain does `utterance` belong to?")):
        for lg in MASSIVE_LANGS:
            yield "%s.%s" % (short, lg), (lambda t=task, i=instr, l=lg: massive(load_dataset, t, i, l))
    for lg in XNLI_LANGS:
        yield "xnli.%s" % lg, (lambda l=lg: xnli(load_dataset, l))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "suites"))
    ap.add_argument("--only", default=None, help="comma-separated suite names, e.g. typed_decisions,en.ag_news")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None
    from datasets import load_dataset
    os.makedirs(args.out, exist_ok=True)
    for name, build in suites(load_dataset):
        if only is not None and name not in only:
            continue
        try:
            rows = list(build())
        except Exception as e:  # one unavailable dataset shouldn't stop the rest
            print("FAIL %-28s %s" % (name, str(e)[:100]))
            continue
        with open(os.path.join(args.out, name + ".jsonl"), "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print("wrote %-28s %4d cases" % (name, len(rows)))


if __name__ == "__main__":
    main()
