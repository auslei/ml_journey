# polish261114 — final polish plan

Plan only. No code is written here. Sections 01 to 07 are on `main`. This feature is the
closing pass: prose quality, length, one weak example, voice consistency, and a clean
full-suite run.

## Measurements (baseline, taken 2026-11-14)

Markdown word counts per notebook (prose only, code excluded):

| Section | Notebook | md-words |
|---|---|---|
| 01 | 01_vectors_and_matrices | 538 |
| 01 | 02_decompositions | 504 |
| 01 | 03_calculus_and_gradients | 597 |
| 01 | 04_autodiff | 588 |
| 01 | intuition_test | 631 |
| 01 | README.md | 284 |
| 02 | 01_probability_basics | 533 |
| 02 | 02_bayes | 551 |
| 02 | 03_probability_in_ml | 523 |
| 02 | intuition_test | 441 |
| 02 | README.md | 273 |
| 05 | 01_neural_net_basics | 595 |
| 05 | 02_backprop_and_autodiff | 667 |
| 05 | 03_optimisers | 593 |
| 05 | 04_learning_rate_schedules | 612 |
| 05 | 05_unstable_gradients | 740 |
| 05 | 06_regularisation_and_callbacks | 714 |
| 05 | 07_hyperparameter_tuning | 615 |
| 05 | 08_pytorch_basics | 705 |
| 05 | 09_cnn_mnist | 720 |
| 05 | 10_transfer_learning | 784 |
| 05 | 11_rnn_and_lstm | 899 |
| 05 | 12_vae | 763 |
| 05 | intuition_test | 1313 |

Em dash counts per section (markdown prose / code cells / README):

| Section | prose em | code em | README em |
|---|---|---|---|
| 01 | 0 | 0 | 0 |
| 02 | 0 | 0 | 0 |
| 03 | 0 | 0 | 0 |
| 04 | 0 | 0 | 0 |
| 05 | 0 | 0 | 0 |
| 06 | 0 | 6 | 0 |
| 07 | 0 | 0 | 0 |

En dash and ` -- ` counts in 01 to 05 prose and READMEs: all 0.

### What the numbers change in the plan

- **Item 4 (strip narrative em dashes, 01 to 05) is already satisfied.** Zero em dashes,
  en dashes, or double hyphens in prose or READMEs across 01 to 05. This drops from a
  rewrite task to a **guard check**: re-run the scan after items 1 to 3 edit prose, and fail
  if any narrative em dash was introduced. The only em dashes in the repo are 6 in section
  06 code cells (print and plot labels), which are out of scope and stay.
- **Item 1 (blog-quality 01 and 02) is a depth pass, not a length pass.** 01 and 02 notebooks
  already sit at 441 to 631 words, at or below the 500 target. "Blog-quality" here means
  better intuition and worked explanation, which tends to *add* words, not cut them. This
  runs in the opposite direction to item 2, so the two must not share a word rule. See Q1.
- **Item 2 (05 to ~500 words) is a trim.** The main-path notebooks run 593 to 899 words, so
  each needs 100 to 400 words cut. `intuition_test` at 1313 is a different format (questions,
  predictions, answers) and is assumed out of the 500 target. See Q2.

## Scope, owners, gates

### Item 3 — better recurrent example in 05 notebook 11 (highest risk, do first)

Current `make_memory_task` fills the gap after step 0 with a constant blank symbol, so the
hidden state is never disturbed. A vanilla RNN carries the one live symbol with no
interference, so it matches or beats the LSTM; the current notebook honestly reports the LSTM
losing. To make the LSTM win reliably we need a task where memory must survive *interference*,
not just time.

- **Fix direction:** add a new seeded generator to `common/sequences.py` (do not change
  `make_memory_task`; other cells and its unit test depend on it). First candidate: noisy
  delayed recall — fill the gap with random non-blank symbols so the state is continuously
  perturbed. Fallback: the adding problem (Hochreiter and Schmidhuber 1997), two marked
  values to sum. Pick by measured result, not preference.
- **Owner files:** `common/sequences.py` (+1 generator), `common/tests/test_sequences.py`
  (+test, both valid output shape and a bad-input raise), `05_deep_learning/11_rnn_and_lstm.ipynb`
  (code + prose rewrite of the training, tuning and "what breaks" cells).
- **Tuning loop:** sweep candidate task x length x epochs across seeds until the LSTM beats the
  vanilla RNN on every seed with a clear mean margin. Quote numbers from a real deterministic
  run (CPU, per the section 07 precedent for determinism).
- **Gate:** new generator test passes; `pytest --nbmake 11_rnn_and_lstm.ipynb` passes; LSTM
  mean accuracy exceeds RNN on every seed by the agreed margin (see Q4); notebook under 30KB
  stripped; prose near the 500 target and em-dash free.

### Item 2 — section 05 notebooks to ~500 words (parallel, excludes nb11)

- **Owner files:** the 11 main-path 05 notebooks except `11` (owned by item 3). `intuition_test`
  excluded pending Q2.
- **How:** cut filler and repetition, keep the maths and the one-run numbers. Numbers are
  quoted from real runs and the notebooks are stored stripped, so any cell whose prose cites a
  figure must be re-executed to confirm the figure still holds after trimming.
- **Gate:** each notebook md-words at roughly 500 (tolerance +/- 75); `--nbmake` passes;
  cited numbers match the fresh run; under 30KB; em-dash free.

### Item 1 — blog-quality prose pass on 01 and 02 (parallel)

- **Owner files:** 01 four notebooks + `intuition_test` + README; 02 three notebooks +
  `intuition_test` + README.
- **How:** deepen intuition and worked steps against `~/dev/my_voice.md` and CLAUDE.md writing
  rules. Australian spelling, short sentences, no AI tells, no em dashes, close each section
  plainly.
- **Gate:** `--nbmake` passes; voice rules met; under 30KB; word ceiling per Q1.

### Item 4 — em dash guard on 01 to 05 (coordinator check, post-edit)

- Already zero at baseline. After items 1 to 3 land, re-run the scan over 01 to 05 prose and
  READMEs. **Gate:** scan returns 0 narrative em dashes. Leave 06 code labels untouched.

### Item 5 — voice and consistency read-through (after items 1 to 4)

Runs last so READMEs describe the final state.

- **Owner files:** all seven section READMEs, root `README.md`, `PROGRESS.md`.
- **Known fixes:**
  - `PROGRESS.md` line 3 and Next-steps 1 still say section 07 is "pending merge" — stale, 07
    is on `main` (see `git log`: PR #2 merged, llm07 docs moved to completed).
  - `PROGRESS.md` "How to resume" still ends with "plan section 07" — replace with the real
    next step after this polish.
  - `PROGRESS.md` Next-steps 2 and 3 describe exactly this feature's work — update to reflect
    it as done.
  - root `README.md` "Future plans" lists the three items this feature delivers — update or
    remove per Q5.
- **Gate:** no stale handoff or roadmap text; voice rules met across all README prose.

### Item 6 — full-suite integration run, all extras (final gate)

```
uv run --extra torch --extra llm --extra nlp pytest
uv run ruff check .
uv run ruff format --check .
UV_PROJECT_ENVIRONMENT=.venv-tf uv run --extra torch --extra tf pytest 05_deep_learning/appendix
```

- **Gate:** all green. Core suite ~6 min; 06 and 07 add a few minutes; the tf appendix runs in
  its own env. Datasets are already cached, so no first-run download cost expected.

## Build order

1. **Item 3** (nb11 + new generator) — start first, longest iteration, empirical.
2. **Item 2** (05 trim, 11 notebooks) and **Item 1** (01/02 depth) — run in parallel with item
   3; independent files, different sections. Item 3's only shared file is `common/sequences.py`
   and it only adds a function.
3. **Item 4** (em dash guard) — coordinator scan once 1 to 3 report done.
4. **Item 5** (voice read-through of all READMEs + root + PROGRESS) — after 1 to 4, so docs
   match final state.
5. **Item 6** (integration run, all extras + tf appendix) — final gate; fix any failures before
   declaring done.

Sub-agents own their notebooks and their tests. The coordinator re-runs the relevant gates
after each sub-agent and the full suite at item 6.

## Risks

- **nb11 may not yield a reliable LSTM win** even with a noisy task; budget iteration and keep
  the adding-problem fallback. If neither gives a clean gap, fall back to reporting honestly at
  a chosen length and raise it in QUESTIONS rather than faking the textbook result.
- **Trimming 05 can break quoted numbers.** Every figure-citing cell must be re-run. CPU vs GPU
  numbers differ by a few points; keep the existing hedge.
- **Item 1 depth vs item 2 trim pull opposite ways.** Keep separate word rules (Q1) so a
  sub-agent does not "trim" 01/02 or "pad" 05.
- **30KB ceiling** on any notebook that gains prose (item 1) — check after editing.

## Expected run time

- nb11 tuning loop: 3 models x 3 to 5 seeds x a few candidate configs, seconds to ~1 min per
  fit on CPU, so roughly 15 to 30 min of experimentation.
- Items 1 and 2 prose edits with re-execution: dominated by notebook run time, a few minutes
  each.
- Integration run: ~6 min core + a few for 06/07 + the tf appendix env.
- Total wall-clock with parallel sub-agents: roughly 1 to 1.5 hours including the nb11 loop.

## Out of scope / untouched

Sections 06 and 07 prose (only their READMEs are read in item 5). The 6 section-06 code em
dashes. Anything under `archive/`, `z_archive/`, and the read-only source backups. No merge or
push without explicit approval.
