# Fixtures

NLaya is tested against the Python `laya` library it ports. `make_fixtures.py` runs the Python
reference and writes:

- `tests/NLaya.Testing/Fixtures/*.json`: golden outputs for tokenization, JSON serialization, prompt
  sequences, language detection and routing, email cleaning, and model outputs (hidden states, logits,
  full `predict` / `predict_batch` results) for all three checkpoints
- `src/NLaya/Lang/lang_data.json`, `src/NLaya/Email/email_data.json`, `src/NLaya/Presets/presets.json`:
  the word lists, script ranges, regexes and presets NLaya embeds instead of re-typing them

Both are generated from the laya commit in `LAYA_COMMIT`. GitHub `main` can be ahead of the PyPI
release even when the version number matches, so always regenerate from that checkout:

```bash
git clone https://github.com/NandhaKishorM/laya && git -C laya checkout $(cut -d" " -f1 tools/fixtures/LAYA_COMMIT)
hf download convaiinnovations/laya && hf download convaiinnovations/laya-multilingual
uv run --python 3.12 --with ./laya --with tokenizers tools/fixtures/make_fixtures.py --laya-tests laya/tests
# a subset: --only tokenizers,sequences,lang,email,presets,models
```

`calibration.json` needs only numpy and torch: it compiles laya's calibration functions out of the
checkout (`ece_score` from `laya/common.py`, `fit_temperature` / `hard_metrics` from the benchmark
notebook builder, `fit_one_temp` from the fine-tuning notebook) without installing laya:

```bash
uv run --python 3.12 --with numpy --with torch tools/fixtures/make_fixtures.py --only calibration --laya-repo laya
```

To move to a newer laya, update `LAYA_COMMIT`, regenerate, and run both test projects: any
behaviour change shows up as a fixture mismatch.
