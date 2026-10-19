# Heart Risk 2026 — interview demo

A Streamlit chat app that interviews a person about their habits, accepts whatever
clinical numbers they happen to have, and then shows how solid the resulting
estimate actually is.

It was built from a 30-minute competition entry ("clean the dataset, make the
models compete") into something presentable. The original scripts are gone; what
replaced them is in `heartrisk/`.

## Read this first

**This is an academic demo trained on synthetic data. It is not a medical device
and it cannot diagnose anything.**

The dataset's own author describes it as "9,000 synthetic patients". The model
learned the rules of a generator, not human biology. Its accuracy figure measures
how well we reverse-engineered that generator, and nothing else. The app says so
on every screen, and the interview stops and gives emergency guidance if someone
describes an acute symptom.

If you have symptoms, see a doctor.

## Quick start

```bash
uv sync                                  # or: pip install -e .
python scripts/fetch_data.py             # downloads and SHA-256 checks the dataset
python -m heartrisk.train                # writes artifacts/model.joblib (~40s)
streamlit run app.py
```

No API key is needed. Without one the interview runs on a deterministic engine;
with `GEMINI_API_KEY` set it runs on Gemini instead, which understands loosely
phrased answers ("fumo desde los 20", "camino poquísimo") that the deterministic
parser would miss.

```bash
export GEMINI_API_KEY=...   # optional; free tier. The app discloses that Google
                            # receives the text, because it does.
```

## Measured performance

From `artifacts/metrics.json`, on a stratified 75/25 split, `random_state=42`:

| Model | Design columns | Threshold | Recall | Precision | PR-AUC |
| --- | --- | --- | --- | --- | --- |
| full contract | 30 | 0.3367 | 0.852 | 0.734 | 0.887 |
| interview-answerable fields only | 19 | 0.2267 | 0.853 | 0.518 | 0.745 |

ROC-AUC for the shipped model is 0.940. Accuracy is deliberately not the headline:
30.3% of rows are positive, so a model that always answers "no" scores 69.7%.

The threshold is chosen for recall, not accuracy, because the expensive mistake in
a screening-shaped product is telling someone they are fine when they are not.
That choice is what closes the recall gap between the two rows above — the
interview-only model reaches the same recall only by flagging far more healthy
people, which is why precision and PR-AUC are the columns to read.

## Why the app shows a confidence band

The strongest predictors in this dataset are not habits. `max_heart_rate_achieved`
comes from a stress test, `st_depression` from an electrocardiogram, and the lipid
panel from a blood draw. A person talking to a chatbot can answer about fourteen
of the twenty-five contract fields; the rest only exist after a measurement.

So `heartrisk/predict.py` never hides an imputation. Every answer carries the count
of real fields, the names of the fields that were filled in from the training
distribution, the measurements the person could still go and get, and a confidence
level. Below 50% coverage it returns no verdict at all, because the honest answer
to a two-answer interview is that there is nothing to say yet.

## Layout

```
heartrisk/
  schema.py     the data contract: fields, plausibility bounds, and FEATURE_ORDER
  data.py       load, validate, clean, build the design matrix
  train.py      fit, evaluate, choose the threshold
  artifact.py   the frozen artifact: what it holds, how it is saved and refused
  predict.py    serve a partial payload and declare the coverage
  interview.py  the Interviewer seam, plus the deterministic implementation
  llm.py        the Gemini implementation and the tool schema
  safety.py     acute-symptom detection
  text.py       Spanish number and text normalisation
app.py          the Streamlit UI
scripts/fetch_data.py
tests/
odd/tasks/heart-risk-chat-demo.md   the specification this was built against
```

Two modules are worth reading first. `schema.py` is the single source of truth
shared by training, serving and the interview — including whether a layperson can
answer a field at all. `predict.py` is where the honesty requirement lives.

## The dataset

Kaggle slug `uditjain13/heart-disease-risk-2026`, version 3, CC0 Public Domain,
9000 rows and 27 columns, downloaded and verified by `scripts/fetch_data.py`.

The copy originally committed to this repository was corrupt: only 6.8% of its
bytes were printable, it held 782,680 NUL bytes, and it contained fragments of
unrelated BMP data. Its byte length matched the genuine release exactly, which is
how we know it was a mangled copy rather than a different file. The fetch script
pins the SHA-256 so the same corruption cannot pass unnoticed again.

Two bugs in the original cleanup are worth naming, because both were silent:

* `sex` was mapped with `1 if x == 2 else 0` against a source column holding
  `"Male"` and `"Female"`, so all 9000 rows became `0` and the feature was dead.
* `st_depression` (the third strongest predictor, |r| = 0.36), `hba1c` and `bmi`
  were dropped while weaker features were kept.

## Development

```bash
.venv/bin/python -m pytest          # 162 tests
```

The contract is enforced from three sides, and there are tests for each:

* `tests/test_data_contract.py` — validation, encoding, and the stability of the
  frozen column layout. A reordered `FEATURE_ORDER` would silently mean the model
  reads the wrong number for every tree in the forest, so it is pinned.
* `tests/test_artifact.py` — the artifact round-trip, stale-artifact refusal, and
  the coverage contract.
* `tests/test_interview.py`, `tests/test_llm.py`, `tests/test_safety.py` — the
  interviewer seam, the tool schema, and the emergency rules.
* `tests/test_app.py` — `streamlit.testing.v1.AppTest` executes the real script
  headlessly: the app starts clean, a bad answer is re-asked rather than stored,
  stopping early refuses a verdict, and a complete 25-turn interview reaches 25/25
  at high confidence.

### Adding a field

1. Add a `Field` to `FIELDS` in `heartrisk/schema.py`, with `askable` and, when it
   is not askable, the `needs_es` that tells the user where to get it.
2. Bump `SCHEMA_VERSION`. Serving code refuses a mismatched artifact rather than
   reading the wrong column, so this is not optional.
3. Retrain: `python -m heartrisk.train`.

The tool schema, the question order and the design matrix all derive from
`FIELDS`, so nothing else needs editing.

## Design tokens

The colours come from `.streamlit/config.toml`, sampled from screenshots of the
Algorithmics course page with `PIL` rather than eyeballed: primary `#833AE0`,
lavender `#F0DAFF`, text `#0F1C2C`, with deep purple `#602B7A`, indigo `#220E56`,
neutral chip `#E8ECF0`, cream `#FFF4E2` and mint `#E6FFE2` named as constants in
`app.py` for tokens the theme cannot express.

The heading typeface is Nunito, which is an **inference** from those screenshots,
not a measured value. Confirm it against the brand manual before shipping.
