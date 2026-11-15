# Open questions: motivation261115

Each has a recommended default. If you are happy with the defaults, I will build to them.

## Q1. Scope, given 43 of 47 openers are already STRONG

The survey found 43 notebooks already open with a clear motivation (sections 01 and 02 were
polished; 03 to 07 open with everyday pictures and concrete examples). Only 4 are
PARTIAL/thin (26_scorecard, 06_evaluating_unsupervised, 07_hyperparameter_tuning, and a light
touch on 09_cnn). A literal "add a motivation section to every notebook" would duplicate
content on the 43.

**Recommended default:** Standardise, do not duplicate. Every notebook gets a consistent
`## Why this matters` heading, but on STRONG notebooks that heading retitles/absorbs the
existing opener (little or no new prose). Real new writing goes into the 4 thin ones plus the
parameter glosses and sample-data comments. This keeps the consistency Anthony wants without
breaking the no-duplication rule.

Alternative: force a fresh, separate motivation block above the existing opener on all 47.
Not recommended; it duplicates and bloats.

## Q2. Word budget and the ~500-word target

`CLAUDE.md` sets ~500 words of prose per notebook. The motivation block is 120 to 200 words.

**Recommended default:** Count motivation **inside** the ~500, not on top. For STRONG
notebooks this is free (the opener already is the motivation). For the 4 thin ones it means
trimming elsewhere if a notebook is already at 500. This protects the 30KB ceiling and the
"prose is short" standard.

Alternative: treat motivation as a separate allowance on top of 500 (up to ~700 total).
Reasonable if you want the motivation to always be additive, but it edges notebooks toward
the size limit and the house style of short prose.

## Q3. How ambitious should the diagrams be?

The survey listed many "diagram opportunities", but the notebooks already plot heavily and
most opportunities are nice-to-haves or already covered. A full sweep risks pushing past
30KB and duplicating existing plots.

**Recommended default:** Minimal. Add a diagram only where a complex idea currently has **no**
visual and the picture clearly helps. That short list: computation-graph schematic (04, 05);
loss-trajectory-per-LR (03); running-mean convergence (02/01); learned-filter grid (09);
trial scatter (07); a first plot each for 06/01 and 06/05 (which currently have none); and a
before/after bar in 07. Everything else stays as-is.

Alternative: build every opportunity the survey surfaced. Not recommended; high effort,
duplicates existing visuals, pressures the size limit.

## Q4. Where do new plotting functions live?

The prompt suggested `common/plotting.py` or a new `common/diagrams.py`.

**Recommended default:** Extend `common/plotting.py` (currently 82 lines) for the one plot
reused across notebooks (the computation-graph schematic used by 04 and 05). Keep
single-notebook plots as `def`s inside their notebook. Do **not** create
`common/diagrams.py`; there is not enough shared, reusable plotting to justify a second
module, and sections already import from `common.plotting`.

Alternative: create `common/diagrams.py` as a home for all new schematic helpers. Only worth
it if Q3 goes ambitious and we end up with five or more shared diagram functions.

## Q5. Intuition tests and the TensorFlow appendix: in or out?

**Recommended default:** Out, for both.
- Intuition tests (`intuition_test.ipynb`, `regression_intuition_test.ipynb`) are the
  closing predict-then-run exercises; a motivation banner at the top would give away the
  answers.
- The TF appendix lives in a second environment (`.venv-tf`), off the main path, and only
  runs nbmake when TensorFlow is installed.

Alternative: include the TF appendix (it is a real content notebook). Low cost if you want
full coverage, but it needs the `.venv-tf` env to test, which the per-section agents will
not have set up.

## Q6. Heading text

**Recommended default:** `## Why this matters` as the standard heading, placed above the
existing `## 1. Intuition` section (which several notebooks already use).

Alternative: fold motivation into the existing "Intuition" heading rather than adding a
distinct one. Cleaner for notebooks that already have a strong Intuition section, but makes
the feature less visibly consistent across the set. I lean to the distinct heading so every
notebook obviously carries the section.
