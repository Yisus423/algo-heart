# Heart Risk Chat Demo

Feature: turn a 30-minute competition script into a presentable Streamlit + LLM chat demo for the Algorithmics marketing team.
Status: implementing. Tasks T1-T8.
Branch: `feat/heart-risk-chat-demo`. Base: `master` @ dae9e57.

## Specs

Numbered requirements. `S#` quote the user's exact words where the user supplied words.

- **S1** — Context and goal: "el objetivo era limpiar el dataset de heart_disease_risk_2026.csv y llevar a los modelos a competir. Yo gané y ahora el profe quiere mostrarles esto a los de marketing." The artifact is a demonstration, not a medical product.
- **S2** — "Crearle una UI bonita tipo chat en streamlit para que el usuario introduzca sus datos."
- **S3** — "Usar algún llm (el free tier de un proveedor como google gemini) para que 'entreviste' al usuario y cuando tenga la informacion lista el modelo pueda hacerle una tool call a una función predict con esos datos."
- **S4** — The problem the interview must solve: "solucionando una problemática tan especifica como que alguien no sepa los números pero sí sus habitos."
- **S5** — Visual reference: style/colorscheme taken from the Algorithmics course page (`learn.alg.academy/login`). Tokens were sampled from three screenshots the user supplied in `examples-style/`.
- **S6** — Safety constraint, stated as the user's dominant fear: "tengo un miedo tremendo, imagináte que un proyecto estúpido le diga a alguien que puede tener fallas en el corazón". The app must never assert that a person may have a heart condition.
- **S7** — Provenance: "estoy leyendo en la descripción de kaggle y dice que es un dataset sintético". The Kaggle author's own subtitle is "9,000 synthetic patients: labs, wearables, lifestyle, 30% prevalence". Every user-facing claim about accuracy must be scoped to the synthetic generator, never to clinical validity.
- **S8** — Decision recorded this session: the product promise is the **hybrid with a visible confidence band** (user's selected option). A prediction is shown with the count of real versus imputed fields, and the band must visually outweigh the verdict.

## Measured baseline

Two real measurements that answer different questions. Both are RandomForest on a stratified 75/25 split with `random_state=42`.

At the default 0.5 cutoff, on the full release, feature-set comparison:

| Feature set | Accuracy | Recall | PR-AUC |
| --- | --- | --- | --- |
| 16 kept features (old `heart_clean.csv`) | 87.2% | 71.3% | 0.863 |
| all 26 encoded features | 88.4% | 73.2% | 0.884 |
| habits only (7 columns a layperson can answer) | 77.4% | 48.5% | 0.640 |

With the threshold chosen for the artifact's recall target, from `artifacts/metrics.json`:

| Model | Design columns | Threshold | Recall | Precision | PR-AUC |
| --- | --- | --- | --- | --- | --- |
| full contract | 30 | 0.3367 | 0.852 | 0.734 | 0.887 |
| interview-answerable fields only | 19 | 0.2267 | 0.853 | 0.518 | 0.745 |

The interview-only model reaches the same recall only by flagging far more healthy people: precision 0.52 against 0.73, and PR-AUC 0.745 against 0.887. The threshold is what closes the recall gap, not new information, and the cost is false alarms. That measured gap is what the confidence band is grounded in, instead of a guess.

ROC-AUC for the shipped model is 0.940. Accuracy is not reported as a headline: 30.3% of rows are positive, so answering "no" unconditionally already scores 69.7%.

Consequence for S4/S8: an interview that never asks for a measurement cannot be presented as a screening result. The app has to count real fields and say what is missing.

## Algorithmics design tokens (sampled, not guessed)

| Token | Value | Sampled from |
| --- | --- | --- |
| brand primary | `#833AE0` | gallery CTA fill, chip text, login link |
| brand primary deep | `#602B7A` | logo banner |
| indigo surface | `#220E56` | illustration blocks |
| lavender tint | `#F0DAFF` | technology chips, secondary buttons |
| primary button tint | `#C19CEF` | login submit fill |
| text | `#0F1C2C` | headings and labels on all three screens |
| surface | `#FFFFFF` | header, cards |
| surface alt | `#F7F9FA` | cards |
| chip neutral | `#E8ECF0` | level chips |
| accent cream | `#FFF4E2` | "Objetivos y logros" section |
| accent mint | `#E6FFE2` | next-class card |
| gradient | `#8355D5` to `#AF78F1` | login right panel |

Heading typeface is a heavy geometric sans (Nunito/Montserrat family). This is an inference from the screenshots, not a measured value; confirm against the brand manual before shipping.

## Tasks

| ID | Specs | Task | Route | Commit |
| --- | --- | --- | --- | --- |
| T1a | S1, S7 | Verified dataset plus a contract-checked cleaning pipeline: pinned download, `sex` fix, evidence-based feature set, frozen `FEATURE_ORDER` | inline, test-first | `1fa6bee` (data swap), this commit (logic) |
| T1b | S1, S7, S8 | Frozen artifact: fit, evaluate, serialise model + imputation + threshold + schema version, and a serving function that declares its coverage | inline, test-first | pending |
| T2 | S2, S5 | Streamlit shell with the Algorithmics tokens in `.streamlit/config.toml` plus minimal CSS | inline | pending |
| T3 | S2 | Chat UI: `st.chat_message` / `st.chat_input`, session state, message history | inline | pending |
| T4 | S3, S4 | LLM interview behind a provider interface, strict tool schema, `predict` tool call | inline | pending |
| T5 | S4, S8 | Coverage and confidence band: nullable fields, real-vs-imputed count, refusal when coverage is insufficient, user confirmation before predicting | inline | pending |
| T6 | S6 | Safety guardrails: emergency routing, persistent synthetic-demo disclaimer, no diagnosis wording | inline | pending |
| T7 | S1, S6 | README and runbook, including what the model is and is not | inline | pending |
| T8 | all | Verification: focused tests plus a whole-document read against each `S#` | inline | pending |

Routing note: T2 depends on T1's artifact, T4 and T5 depend on T2's shell, so these units are not independent and the Writer rule does not fire. All tasks run inline in the parent session.

## Delivery slices

The review slice metric counts authored changed lines, and the dataset swap alone is 9001 of them. It is mechanical, so it was split into its own commit (`1fa6bee`, data swap only, no logic) to keep the reviewable slice about the code. After that split the feature's reviewable surface is roughly 950 authored lines across 5 files, delivered as: data swap, then the contract and pipeline, then the artifact and serving layer, then the app and the interview.

## Log

**L1 (user, verbatim):** "mira este proyecto que tengo aquí, fue hecho en una competencia de media hora (está horrible), el objetivo era limpiar el dataset de heart_disease_risk_2026.csv y llevar a los modelos a competir. Yo gané y ahora el profe quiere mostrarles esto a los de marketing. Tenía pensado en esto:
- Crearle una UI bonita tipo chat en streamlit para que el usuario introduzca sus datos.
- Usar algún llm (el free tier de un proveedor como google gemini) para que "entreviste" al usuario y cuando tenga la informacion lista el modelo pueda hacerle una tool call a una función predict con esos datos, solucionando una problemática tan especifica como que alguien no sepa los números pero sí sus habitos.
Antes de escupir código cómo locos me gustaría estudiar y ver primero, me gustaría ver si podría conectarte el mcp de playwright para que sacaras el estilo/colorscheme de la página del curso (learn.alg.academy/login) como referencia o si sabes, decirme una manera mejor de hacer esto.
Aparte tengo un miedo tremendo, imagináte que un proyecto estúpido le diga a alguien que puede tener fallas en el corazón, así que luego de terminar podríamos enfocarnos en el classifier y el dataclean (aunque ahora estoy leyendo en la descripción de kaggle y dice que es un dataset sintético, mucho más miedo que antes)"

**L2 (user, verbatim):** "usa codegraph" — codegraph was used; it reported 8 symbols across `model_heart.py` and `main.py`, 1 caller each, and no tests within 3 caller hops.

**L3 (user, verbatim):** "sí, creo que el original está corrupt y mi profesor lo subió a drive así, luego lo descargo de kaggle ." — confirms the corrupt-file hypothesis and that the Kaggle download is the replacement path.

**L4 (user decisions, this session):** product promise = "Híbrido con banda de confianza visible"; style source = "Me pegás una captura y la leo". The user then supplied three screenshots in `examples-style/`.

**L5 (evidence, verified this session):**
1. `heart_disease_risk_2026.csv` is corrupt: 1137455 bytes, only 6.8% printable, 782680 NUL bytes, contains `.bmp` filenames. Byte length is identical to the genuine Kaggle CSV, so it is a mangled copy, not a different file.
2. Genuine source: Kaggle slug `uditjain13/heart-disease-risk-2026`, version 3, CC0 Public Domain, no authentication required. The archive is a ZIP containing `heart_disease_risk_2026.csv` (1137455 bytes, sha256 prefix `f30b84b747c1d263`), 9000 rows x 27 columns.
3. `heart_clean.csv` target distribution (6273/2727) matches the genuine v3 data, so the existing cleaned file is valid data.
4. `sex` mapping bug: the source column is a **string** (`"Male"`/`"Female"`), and `heart_dataclean.py` maps `1 if x == 2 else 0`, so all 9000 rows became `0`. The column is dead.
5. Column drops were not evidence-based: dropped `st_depression` (|corr| 0.357, third strongest), `hba1c` (0.226), `bmi` (0.213) while keeping `chest_pain_type` (0.116), `triglycerides` (0.110), `family_history` (0.071).
6. Baseline comparison (RandomForest, 300 trees, stratified 75/25, `random_state=42`): figures in the table above.
7. `model_heart.py` never serializes the model or the `StandardScaler` and has module-level training side effects on import, so nothing was servable.
8. Environment: uv venv at `.venv`, pandas 3.0.6, scikit-learn 1.9.1, streamlit 1.66.0, joblib present; no LLM SDK installed.

**L6 (decision, this session):** work order is pipeline, then data contract, then app, then LLM interview. The user originally proposed UI first; the app cannot be built before the frozen artifact exists, so the order was corrected and accepted.

**L7 (convention, this session):** user-facing UI copy and the app's messages are Spanish, because the audience and the referenced course interface are Spanish. Code, identifiers, comments, commit messages and this document are English. The previous script's Spanish comments stay as they are until the file is rewritten.

**L8 (defect found this session, deferred):** `model_heart.py` prints confusion-matrix labels describing mortality and survival ("murió y se predijo muerte") for a target that is actually `has_heart_disease`. Misleading in a health demo; corrected by T1.

**L9 (measurement, this session):** the pandas in this environment is 3.0.6, where string columns have dtype `str` rather than `object`. `select_dtypes("object")` still matches them but is deprecated. The contract uses `is_string_dtype` and explicit `"str"` selection so the behaviour is not accidental.

**L10 (differential check, this session):** the new pipeline was compared against the base commit's `heart_clean.csv`, regenerated from the base `heart_dataclean.py` in a scratch directory. Of the 16 shared columns, 15 are identical row for row and only `sex` differs, in exactly the 4731 rows the old mapping had zeroed. No unintended behaviour change.

**L11 (slice decision, this session):** the harness flagged the first commit at 19153 authored changed lines, almost all of it the dataset file. The commit was split so the mechanical data swap stands alone; see Delivery slices.

**L12 (bug found by a smoke test, this session):** `python -m heartrisk.train` wrote an artifact that could not be loaded. Running a module with `-m` executes it as `__main__`, so a dataclass defined in `heartrisk/train.py` was pickled as `__main__.Artifact` and `import heartrisk` could never resolve it. Fixed by moving `Artifact` into `heartrisk/artifact.py`, a module that is never `__main__`, and pinned by `test_the_artifact_class_is_importable_and_not_main` plus a subprocess test that runs the real CLI and reloads what it wrote. This would have failed on stage, not in review.

**L13 (measurement, this session):** my own earlier framing was incomplete. The 48.5% figure for habits-only was measured at the default 0.5 cutoff. At a recall-matched threshold the interview-only model reaches 0.853 recall, so "a conversation misses half the cases" was wrong as stated. The real cost is precision: 0.518 against 0.734, with PR-AUC 0.745 against 0.887. The corrected comparison is in Measured baseline.

**L14 (size, this session):** the artifact is a 300-tree forest, about 31 MB uncompressed. `joblib` compression level 3 brings it to 6.2 MB, so it stays out of git and is rebuilt in about 40 seconds.
