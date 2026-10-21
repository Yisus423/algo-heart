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
- **S9** — Language of the copy, corrected by the user: "dentro de la app hay que cambiar el texto, usaste el español argentino de tu persona en algo que deberia ser neutro, sin voseo ni nada, sería super raro llevar eso asi estando en venezuela". Every user-facing string is neutral Spanish: no voseo, no Rioplatense lexicon, no Argentina-only emergency numbers. Understanding the user's input stays dialect-tolerant.
- **S10** — Credentials, corrected by the user: "la api deberia usarse mediante secrets para que los usuarios puedan entrar y probar sin poner API keys". The deployment owns the key and supplies it through Streamlit secrets, so a visitor never has to bring one. A missing key must be loud rather than a silent downgrade.
- **S11** — Visual quality, deferred by the user: "la web te quedó re fea y con errores de diseño, dejemos eso de último". The redesign is tracked and deliberately last, after the behaviour is right.

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

## Deviations

Recorded explicitly rather than quietly absorbed.

- **S3, who decides the moment to predict.** S3 has the model call a `predict` function "cuando tenga la informacion lista". Here the model calls a tool that *reports what it understood*, and the app decides when to predict, from the coverage gate in `predict.py`. Reason: letting the model decide readiness puts the "is this enough to say something" judgement inside the component that has no knowledge of coverage, recall or the synthetic-data caveat, which is precisely the failure S6 exists to prevent. The user's original wording is preserved in S3; if the model owning that decision matters, this is where to say so.
- **Heading typeface.** Nunito is inferred from the screenshots. It is marked as an inference in `app.py` and in the README rather than presented as a measured token.

## Tasks

| ID | Specs | Task | Route | Commit |
| --- | --- | --- | --- | --- |
| T1a | S1, S7 | Verified dataset plus a contract-checked cleaning pipeline: pinned download, `sex` fix, evidence-based feature set, frozen `FEATURE_ORDER` | inline, test-first | `1fa6bee` (data swap), `49c5985` (logic) |
| T1b | S1, S7, S8 | Frozen artifact: fit, evaluate, serialise model + imputation + threshold + schema version, and a serving function that declares its coverage | inline, test-first | `115b2a5` |
| T2+T3 | S2, S5, S8 | Streamlit shell with the Algorithmics tokens, chat UI with session state, the coverage band, and a deterministic stub interviewer behind the interface a language model will implement | inline; the delegate was unavailable | this commit |
| T4 | S3, S4 | Gemini interviewer behind the same interface, with the answer extracted through a strict tool call | inline, test-first | this commit |
| T5 | S8 | Coverage band | dropped: absorbed by T2+T3 | — |
| T6 | S6 | Emergency routing and the persistent disclaimer | inline, test-first | this commit |
| T7 | S1, S6 | README and runbook, including what the model is and is not | inline | `1c2b3c8` |
| T8 | all | Writer verification: focused tests plus a whole-document read against each `S#` | inline | `1c2b3c8` |
| T9 | S3 | Live Gemini call, to replace the `FakeTransport` evidence with a real one | blocked: no `GEMINI_API_KEY` was provided | — |
| T10 | all | Independent verification required by the escalated plan | blocked: no subagent runtime available, see L26 | — |
| T11 | S9 | Neutral Spanish across every user-facing string, including the interviewer's questions, the safety message and the model's system prompt | inline, test-first | `cabe47d` |
| T12 | S10 | The key comes from Streamlit secrets so a visitor brings none, and a missing key is loud rather than a silent downgrade | inline, test-first | this commit |
| T13 | S11 | Visual redesign pass: layout, spacing and the design errors the user flagged | pending, deliberately last per the user | — |

T5 was planned as a separate task and turned out to be presentation logic that only makes sense next to the chat that produces the payload, so it was folded into T2+T3 rather than left pending.

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

**L15 (delegate unavailable, this session):** T2+T3 was delegated to `gentle-ai-worker` in the background. It timed out after four minutes having emitted no first run event for its model (`nvidia/deepseek-ai/deepseek-v4.1-flash`). Reported as unavailable and done inline. The first launch attempt was also rejected for a malformed `## Allowed edit surfaces` block, which the writer validates strictly: one narrow path or glob per line, with no prose in that section.

**L16 (decisions, this session):** provider for T4 is Gemini on the free tier via `google-genai`, which requires `GEMINI_API_KEY`. Strictness stays as specified: a strong permanent disclaimer, and the coverage floor stays at 0.5, so a complete interview earns a verdict only at the lowest confidence level.

**L17 (two parser bugs found by their own tests, this session):** the answer parser had bugs that the tests caught immediately and that would have produced wrong data silently. (1) The number regex matched digits inside words, so "mi hba1c es 5,4" parsed as 1.0 from the `1` in `hba1c`. (2) Single-word synonyms matched as substrings, so the `"m"` cue matched inside `"mujer"` and every woman was recorded as male. Both are now pinned by tests. The lesson worth keeping: a cue that matches a substring of another word is a data-corruption bug, not a parsing annoyance.

**L18 (app verification, this session):** the app is verified by `streamlit.testing.v1.AppTest`, which executes the real script headlessly: the app starts without raising, answering advances the conversation, an unusable answer is re-asked rather than stored, stopping early shows the band with no verdict, a complete 25-turn interview reaches 25/25 at high confidence with a verdict, the verdict never claims a diagnosis, and the interview restarts. A real server start was also checked (`streamlit run app.py` served HTTP 200 on both `/healthz` and `/`).

**L19 (scope boundary, this session):** ambient requests to expose the model under a real product surface stay out of scope. The app is a demo for one classroom presentation, not a clinical tool, and S6 is the reason.

**L20 (safety false positive, caught by a test, this session):** the first version of the emergency rules listed the bare word "infarto" as acute. That fired on "mi viejo tuvo un infarto a los 55", which is the *answer* to the family-history question: the interview stopped and told the user to call an ambulance. Fixed by splitting the rules in two: only unambiguous acute presentations fire alone, and ambiguous or ordinary terms ("infarto", "arritmia", "taquicardia", "me falta el aire", "convulsiones") need a present-tense marker. Pinned by `test_describing_a_relative_heart_attack_is_not_an_emergency` and at app level by `test_describing_a_relative_who_had_a_heart_attack_is_not_an_emergency`.

**L21 (architecture decision, this session):** in the language-model interviewer the model owns *understanding* and the contract owns the *agenda*. The model reads free text and reports what it understood through the tool call; the next question always comes from `FIELDS` order via `InterviewState.next_field()`. That means the model cannot loop, skip a field, invent a question, or strand the interview. Two fallbacks keep it demo-safe: the first turn is deterministic, so nothing blocks on the network before the user types; and the deterministic parser backs up the field currently being asked. The tool schema is generated from `FIELDS`, so a field can never be askable yet unreportable.

**L22 (API verified, not assumed, this session):** the `google-genai` 2.31.0 surface was confirmed by introspection before being used: `genai.Client(api_key=...)`, `client.models.generate_content(model=, contents=, config=...)`, `types.GenerateContentConfig(system_instruction=, tools=, temperature=)`, `types.Tool(function_declarations=)`, `types.FunctionDeclaration(name=, description=, parameters_json_schema=)`, `response.function_calls`, `response.text`.

**L23 (verification boundary, this session):** T4 is not verified against the live API, because `GEMINI_API_KEY` was not provided. The wire path is covered by a `FakeTransport` that exercises extraction, validation, rejection reporting, fallbacks and the emergency short-circuit without a network, and the app falls back to the deterministic interviewer when no key is present. The free tier's data use is disclosed in the UI, since the user chose that provider.

**L24 (test determinism, this session):** the app tests delete `GEMINI_API_KEY` with an autouse fixture, so a machine that happens to export the variable cannot turn the suite into live API calls that spend quota and depend on the network.

**L25 (verification, this session).** Each spec, against evidence:

| Spec | Evidence |
| --- | --- |
| S1 | The artifact is a demonstration: the header disclaimer, the README's "Read this first", and the verdict wording all say so. |
| S2 | `app.py` uses `st.chat_message` / `st.chat_input` with `session_state`; `tests/test_app.py` drives a full conversation through the real script. |
| S3 | Implemented with the documented deviation in Deviations: Gemini understands the answers, the contract owns the agenda, the tool schema is generated from `FIELDS`. Not verified against the live API, see L23. |
| S4 | The interview asks all 25 fields, phrases the 11 measurement ones with `needs_es`, and the band reports what is still missing. `test_the_band_names_what_the_user_could_go_and_get_measured`. |
| S5 | Tokens sampled from the screenshots with PIL, in `.streamlit/config.toml` and named in `app.py`. |
| S6 | Permanent disclaimer (`test_the_synthetic_data_disclaimer_is_visible`), verdict that never claims a diagnosis (`test_the_verdict_never_claims_a_diagnosis`), emergency routing (`tests/test_safety.py`, plus the app-level emergency tests), and the coverage gate below which no verdict exists. |
| S7 | The disclaimer and the README both name the synthetic provenance. |
| S8 | `render_band` puts coverage and confidence first and full width; the verdict is a caption after a divider. `test_stopping_early_shows_the_band_and_refuses_a_verdict`, `test_a_complete_interview_reaches_full_coverage_and_a_verdict`. |

Checks actually run: 162 tests in the working tree; the documented quick start on a **clean clone** (fetch verifies the committed CSV, train writes the artifact, 162 tests pass); `streamlit run app.py` served HTTP 200 on `/healthz` in both the working tree and the clone; and the differential check in L10. Not run: a live Gemini call, because no `GEMINI_API_KEY` was provided.

**L26 (independent verification unavailable, this session):** the escalated plan required an independent verifier, so `gentle-ai-verify` was launched. It timed out with the identical symptom as the earlier writer: no first run event for `nvidia/deepseek-ai/deepseek-v4.1-flash` after four minutes. Two different agents, the same model, the same stall pattern, so subagent execution is unavailable in this environment rather than the task being wrong. It is reported as unavailable and was not retried or replaced with ceremony. **The independent verification therefore remains open, and the writer self-verification stands alone.**

The writer's own verification of the claim most likely to be self-deceived - that no user-facing string asserts a heart condition - was done mechanically rather than by re-reading my own prose. Every string literal in `app.py`, `heartrisk/llm.py`, `heartrisk/interview.py`, `heartrisk/safety.py` and `heartrisk/predict.py` was extracted with `ast` (121 strings) and searched for condition-asserting cues. Seven hits, all benign on inspection: the disclaimer points the reader at a doctor rather than at a condition, "si los tienes" refers to numbers, and the system prompt carries its own prohibition. That is evidence, not proof: an independent reader could still find an implication I cannot see in my own text.

**L27 (user correction, verbatim):** "por otro lado tambien hay cosas que no me gustaron y que hay que refactorizar, por ejemplo, la api deberia usarse mediante secrets para que los usuarios puedan entrar y probar sin poner API keys, literalmente no existía la posibilidad dentro del programa que no hubiera API key. Aparte la web te quedó re fea y con errores de diseño, dejemos eso de último"

**L28 (user correction, verbatim):** "dejalo en inglés, pero dentro de la app hay que cambiar el texto, usaste el español argentino de tu persona en algo que deberia ser neutro, sin voseo ni nada, sería super raro llevar eso asi estando en venezuela"

**L29 (requirement change, this session):** L27 and L28 add S9, S10 and S11 and reopen nothing else. S9 corrects my own convention decision in L7: I had written the UI copy in the persona's Rioplatense Spanish for a Venezuelan audience, which is a product defect and not a stylistic preference. S10 changes where the credential comes from and how its absence behaves. S11 is deferred to last by the user's own instruction.

**L30 (open question, this session):** the previous safety message named "107 en CABA" and "911 en el resto del país", which is an Argentina-only split. S9 requires a neutral form; the replacement says "en Venezuela, 911", and that number needs the user's confirmation rather than my assumption, because a wrong emergency number is the one string in this codebase that must not be wrong.

**L31 (T11 and T12 delivered, this session):** T11 removed the voseo from every user-facing string and added `tests/test_copy.py`, which extracts those strings with `ast` and fails on any voseo form or regional word. Writing that check exposed a bug in the check itself: voseo and the neutral second person differ only by the accent ("pesás" from "pesas", "hacés" from "haces"), so a character class like `pes[aá]s` flagged the correct neutral form, and the patterns now spell the accented forms out. T12 moved the credential to Streamlit secrets so the deployment supplies it once and a visitor brings nothing, and made its absence loud instead of a silent downgrade. Input parsing was deliberately left dialect-tolerant: the keyword cues are accent-stripped stems, so a person writing in any register is understood.

**L32 (live Gemini verification, finally run, this session):** with the key the user put in `.streamlit/secrets.toml`, the real path was exercised. Model `gemini-3.1-flash-lite`. What it proved: the tool schema is accepted by the API (a bad `parameters_json_schema` would have returned 400 rather than a function call); from one messy message — "mirá tengo como 62 pirulos, mi viejo tuvo un infarto a los 55, fumo desde los 20 y camino poquísimo" — the model registered `age`, `family_history` and `smoker_status`, and correctly **did not invent** `daily_steps` from "poquísimo" because no number was given; and the emergency detector did **not** fire on the family-history sentence, so the L20 fix holds under real input. A second message proved the guardrail end to end: the model reported `cholesterol_total = 900`, `validate_value` refused it, nothing was stored, and the user was told why.

**L33 (five defects the live run exposed, this session):** reading the real conversation, instead of green tests, found five things that all looked like "the web is ugly":

1. **The greeting was printed twice.** `start_interview` pre-filled `messages` from the opening turn and then called `apply_turn`, which appends the same turn. Fixed by making `apply_turn` the only writer. The tests missed it because they asserted `in`, never a count.
2. **The model asked the same question the app asked next**, so the screen showed "Entendido. ¿Cuál es su sexo?" immediately followed by "¿Cuál es tu sexo?". The system prompt *told* it to formulate the question, while the app appends the contract's own. The prompt now forbids questions outright.
3. **The model addressed the person as "usted"** while the contract uses "tú", because the prompt only banned voseo. It now names both.
4. **Gemini returns no text alongside a tool call**, so banning questions made the interviewer silent. Asking it for a sentence did not work either: the field came back empty on every live run. The acknowledgement is now the app's, chosen from what was actually registered — `ACKNOWLEDGEMENT` — which also keeps the model's one job pure.
5. **A developer message reached the screen.** The rejection read "... así que no los guardé: cholesterol_total: 900 is above the plausible maximum 700": half Spanish, half `SchemaError`. `rejection_message()` renders the same rule in neutral Spanish for the person who typed the value, and the contract's own errors stay English for whoever maintains the pipeline.

**L34 (the suite stopped being hermetic, this session):** moving the credential into `secrets.toml` made the app tests call the real API. The fixture deleted `GEMINI_API_KEY` from the environment, which was enough before and is not enough now, because the app reads secrets first by design. Symptom: the suite went from 80s to 298s, nine tests failed, and the failures were network-dependent. Fixed by neutralizing the secrets source in the fixture while still honouring the environment variable, so the one test about a configured key keeps exercising the real path. Worth remembering: a test that is hermetic by accident stops being hermetic the moment production reads a new credential source.

**L35 (my mistake, corrected by the user, this session):** I installed Playwright as a **project** dev dependency to look at the UI. The user corrected it: "es una herramienta para ti, no para el proyecto". It was removed from `pyproject.toml`, and configured instead as a user-level MCP server (`~/.pi/agent/mcp.json`, `npx -y @playwright/mcp@latest`, 25 tools, `--headless --isolated --viewport-size 1440x900`), alongside codegraph, context7 and engram. The browser cache lives in `~/.cache/ms-playwright`, outside the repo. The same correction applies to how I was reading docs: `context7` was already available and I was introspecting packages by hand instead; the transport was confirmed against the official docs and the `[[theme.fontFaces]]` shape came from them too.

**L36 (the icon diagnosis, and the fix, minus visual confirmation):** the user reported that "las imágenes" were not loading. There are no images in the app, so the suspect was the icon font. Streamlit 1.66 ships `MaterialSymbols-Rounded.D_GK7ERd.woff2` **locally** and applies it through the `stIconMaterial` test id, so nothing depends on a CDN — which ruled out the user's network. The cause was mine: `FONT_STYLE` injected `html, body, [class*="st-"] { font-family: 'Nunito' }` with `unsafe_allow_html=True`, and Streamlit's own elements carry runtime-generated `st-emotion-cache-*` classes, so that selector reached the icon elements and replaced their font. Replaced with native theming: `font`, `headingFont` and `[[theme.fontFaces]]` in `.streamlit/config.toml`, using the variable-font pair Streamlit's own docs prescribe, one file per style covering `weight = "200 1000"`. `unsafe_allow_html` no longer appears anywhere in the app, and `tests/test_ui.py` fails if a broad selector or a `font-family` returns. **Not yet confirmed on screen:** the vision step needs the MCP server, which needs a `/reload`.

**L37 (the visual pass, finally done with my own eyes, this session):** the user moved Playwright into the project's `.pi/mcp.json` and confirmed the icons now render. The MCP could not launch a browser because its default is the `chrome` channel and only the bundled Chromium is installed, so screenshots were taken with `npx playwright screenshot`, which defaults to that bundled Chromium. The first capture showed only Streamlit's loading skeleton, so the wait is on a selector (`[data-testid="stChatInput"]`), never a fixed timeout. What the screenshots showed, and what a green suite could not:

1. **Each line of a turn was its own bubble with its own avatar.** The opening was three separate assistant bubbles, and with the model active every turn added two more, so the history read as a list of unrelated one-liners instead of a conversation. That is what the user meant by "el usuario debería poder ver sus mensajes y respuestas anteriores". A turn's lines are now joined into one bubble.
2. **Two full-width coloured banners stood before any content**: the red disclaimer and a blue "Google receives your text" box. The privacy disclosure is now part of the engine caption, so it stays visible without burying the interview.
3. **The assistant avatar was Streamlit's default orange icon**, fighting the purple palette. It is now an emoji.

**L38 (spec S12, the user's correction to my own prompt fix, this session):** the user rejected the earlier solution outright: "no es que el modelo no haga preguntas sino que haga las que el contrato le diga". They are right, and the better design is the opposite of what I built. The contract still decides *which* field is asked, because that is what stops the model wandering, but the model now decides *how* to ask it. The order is deterministic, so the expected next field is computed **before** the model call and handed over as a per-turn system instruction; the model's reply is used as the question when it matches the field that actually came next and contains a question mark, and otherwise the contract's own wording goes in. No second call, no way for the interview to stall, and a model that only acknowledges cannot leave the person with nothing to answer.

**L39 (Streamlit promotes secrets into the environment, this session):** with a real `secrets.toml` present, the app tests began calling the live API even though the fixture deleted `GEMINI_API_KEY`. The cause is inside Streamlit: `Secrets._maybe_set_environment_variable` runs `os.environ[k] = str(v)` (`.venv/.../streamlit/runtime/secrets.py:429`) when `st.secrets` is read, and the app reads `secret_map()` one line before it builds the interviewer. Deleting the variable was useless: the key was back in the environment a step later and the next turn spent real quota. The fixture now empties `Secrets._parse`, which stops both the leak and the promotion. Worth knowing in production too: after the first render the key sits in the process environment.

**L40 (open design question, this session):** with the banners reduced, the red disclaimer is now the loudest element on the screen, and it is the first thing a visitor sees. It is the safest choice and the least on-brand one. Softening it into a card in the reference palette would look far closer to the brief, and the question of whether a safety notice may stop being alarm-red is the user's to answer, not mine.

**L41 (S12 needs two calls, and the first attempt asked the wrong thing, this session):** the single-call design in L38 cannot work, and the raw response proves why: asked with a tool available, Gemini's reply contains exactly one part, the `function_call`, with `text=None`. There is no text to phrase the question with. The phrasing is therefore its own second call per turn, with `tool=None` so it cannot call anything, which is what suppresses the text. Measured: two calls per turn, the second without tools, and the question that comes back is the model's own wording rather than the contract's.

The first version of that call then produced nonsense: sending the bare question as a user turn made the model read it as addressed to itself, so asked to rephrase "¿Cuál es tu sexo?" it answered "¿Cómo te gustaría que me identificara o qué te gustaría saber sobre mi naturaleza?". The instruction now says explicitly that it is a question to ask of the person and must not be answered, and the tests assert both halves of that instruction are present.

**L42 (resilience, from a real 503, this session):** a live verification run hit "This model is currently experiencing high demand. Spikes in demand are usually temporary." and the app's exception propagated to the screen. On stage that is a traceback in front of an audience, so a failed model call now degrades instead of crashing: the deterministic parser still handles the field being asked, so the answer is not lost, the person is told plainly that the model is unreachable, and the interview keeps moving. `_phrase` degrades the same way, back to the contract's own wording.

**L43 (S14 and S6, to the user's two decisions, this session):** the checklist went exactly where the user proposed — a list that ticks itself — and the disclaimer stopped being a paragraph. Three things were only visible on screen:

1. **The checklist in the main column pushed the chat below the fold.** It took most of the viewport, which defeats the point of a chat. It now lives in the sidebar, where it stays visible and ticks itself while the conversation keeps the room.
2. **The markers were doubled**: `- ○ Edad` rendered as a bullet *plus* my own glyph. The fix is the markdown task-list syntax, which Streamlit renders as real checkboxes — the checkbox the user had asked for in the first place, arrived at from a rendering bug.
3. **`secondaryBackgroundColor` paints the whole sidebar**, not just accents. The `#F0DAFF` sampled from the reference chips became a 300px block of saturated lavender that dominated the page. It is now `#F7F1FE`, with the sampled value still named in `app.py` for real accents.

Grouping by `needs_es` also read as a sentence fragment — "un tensiómetro — ○ Presión sistólica en reposo" — so the contract grew a `source_es` short heading, and a test fails if a measurement field does not declare one. The short heading lives in the contract rather than in the app so a new field cannot ship without it.

**L44 (a Streamlit gotcha that cost a screenshot, this session):** Streamlit hot-reloads the entry script but **not the modules it imports**, so a change under `heartrisk/` needs a server restart. A capture failed with no chat input because the running process still had the old `heartrisk.schema` in `sys.modules`, and `source_group` did not exist in it. Worth remembering before concluding that a change broke the app.

**L45 (the user's sharpest critique, and the measurement that answers it, this session):** "la verdad es que veo cómo algo muy tedioso rellenar los 25 campos respuesta por respuesta". They are right, and it is a defect of the interview design rather than of the visuals: 25 answers, one per turn, is a questionnaire wearing a chat's clothes.

Measured before deciding anything, with a 300-tree forest's feature importances on the full release and a top-k curve read at the same recall target:

| fields | recall | precision | PR-AUC |
| --- | --- | --- | --- |
| top 4 | 0.850 | 0.616 | 0.825 |
| top 6 | 0.850 | 0.701 | 0.870 |
| **top 8** | 0.850 | **0.719** | **0.880** |
| top 14 | 0.850 | 0.726 | 0.883 |
| all 25 | 0.852 | 0.734 | 0.887 |
| interview answers only | 0.853 | 0.518 | 0.745 |

Eight fields carry 98% of what twenty-five do. Worse, three questions contribute almost nothing: `sex` 0.61%, `family_history` 0.48%, `wearable_owner` 0.35% — the app was asking whether someone wears a smartwatch to gain a third of a percent. The first version of this measurement was also wrong: it picked the loosest threshold instead of the strictest, so every row read `recall 1.000`. Only the PR-AUC column was usable, and the run was redone with `train.choose_threshold`, the same function the artifact uses.

The user chose: a free-form opening, then only what is missing, asked in batches.

**L46 (the decision's consequence, flagged before implementing):** the coverage gate is defined over 25 fields, so asking 8 would put a complete interview at 32% and the system would refuse to give a verdict. Coverage has to be measured over the fields the interview actually chases. The band is more honest that way in any case: today it treats a field worth 24% and a field worth 0.35% as equally valuable.

**L47 (the redesigned interview, this session):** the flow the user chose is in. Verified live: the free-form opening, then one messy paragraph — "tengo 62, mi viejo tuvo un infarto, fumo desde los 20, peso 95 y mido 1,72, camino como 3000 pasos y hago unos 20 minutos de ejercicio por semana" — captured **five fields** in one message, and the next ask was a batch of two, phrased by the model as a single message. The core needs about four turns instead of twenty-five. The sidebar went from twenty-five rows to eight checkboxes in four groups.

Two real bugs surfaced while implementing, both caught by tests rather than by review:

1. **`next_field` iterated the importance order**, so after `age` the interview asked for a maximum heart rate from a stress test. Importance is a measure of value, not a sane order to ask in; the order is now the core questions from memory with `age` first, then the core measurements.
2. **The degraded path lost the person's answer.** `_offline_turn` asked a batch, but with the free-form opening there is no pending field, so when the model failed the deterministic parser had nothing to try against and the message was dropped. It asks one question at a time now, because only the model can match several free-text answers to several questions; guessing wrong there files a number under the wrong question.

A batch also carried "si no lo sabes, dime no sé" after *every* question, which turned one ask into three apologies. It carries it once, at the end.

Tradeoffs, stated rather than hidden: two model calls per turn (extraction plus phrasing), and the free-form opening depends on the person writing something. Someone who answers "hola" simply receives the first batch of questions, which is the old behaviour.

**L48 (the measured answer to a design complaint):** the interview was shortened because the user called it tedious, and the numbers say the shortening is nearly free: eight fields give precision 0.719 against 0.734 for all twenty-five. The band now measures coverage over the eight the interview asks for, which is both necessary and more honest — the old denominator counted a field worth 24% and a field worth 0.35% as equal.

**L49 (the result screen and the wait, from the user's screenshots, this session):** four observations, all correct:

1. "la card quedó abajo y se vé descuadrada en posición" — the result card and the header had different widths, and the conversation sat under the result, so the reading order was result, then greeting, then the answers that produced it.
2. "para que el usuario no se desespere al esperar deberíamos hacer que su mensaje se vea directamente en el chat y mostrar un spinner cargando" — Streamlit paints only the deltas it has produced, and the transcript was rendered before the chat handler ran, so the person watched a frozen page for the twenty seconds two model calls take.
3. "mostrar el chat cuando damos el resultado está de más" — once the interview is over the transcript says nothing.
4. "se ve que hacemos más énfasis en los datos recolectados que en el resultado (casi ni se encuentra el resultado)" — the verdict was a grey caption below an expander and a divider, while the loudest element on screen was a bold checklist of answers already given. This is the same class of mistake as the icon font: the element with the most visual weight was not the element that mattered.

What changed: the result screen replaces the conversation, the verdict is a heading with its own name ("Perfil del lado de mayor riesgo") inside the card, coverage and confidence are supporting metrics under a divider, the transcript is one collapsed click away, and the result card shares the header's column so both line up. `st.progress` is gone from the card: inside a bordered container it measured itself against the wrong width and drew 8/8 as two thirds full, which is worse than no bar at all — the metric states the number without lying. The wait now echoes the message immediately and shows a spinner, which is the whole feedback budget of a turn.

Verified by driving the real browser: the echo plus spinner during the wait, and the aligned result screen afterwards.

**L50 (two blocking bugs, from the user's screenshots, this session):** the user looked at the app and found what my screenshots had not, and their closing line was the fair part: "casi todos los bugs que importaron los encontró alguien con ojos, se te olvidó usar playwright para mirar también". I had taken screenshots but not looked at the state after the last change.

1. **"listo" did not end the interview with a model configured.** The `is_done` check existed only in the deterministic interviewer, so the model path re-asked the pending field forever and the result screen was unreachable. I verified it in the screenshot they sent: answering "listo" produced "Perdón, de nuevo: ¿Tienes el valor de frecuencia máxima…". The root cause is not the missing line, it is that two implementations of one interface duplicated the same two checks and one copy drifted. They are now one shared `preflight_turn`, used by both, and a test asserts both paths honour an emergency and "listo" without calling the model.

2. **The input rendered above the person's own message and the spinner.** `st.chat_input` draws at its position in the flow, and I rendered the echo and the spinner *after* it. The answer is worked on in its own script run now, with the input hidden and the spinner in its place.

Verified by driving the browser and looking: the message appears at once with the spinner below it and no stray input, and "listo" reaches the result screen. The layout complaint about the card was almost certainly the same bug seen from outside: they never reached the result screen, so the only card they could judge was the checklist, unchanged.
