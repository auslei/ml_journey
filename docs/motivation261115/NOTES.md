# Build-time notes: motivation261115

Minor assumptions recorded during the build. Major concerns go to QUESTIONS.md under
"Build-time questions" with a BLOCKED stop.

## Phase 0

- Added `draw_computation_graph(nodes, edges, ...)` to `common/plotting.py`. Caller supplies
  node positions so the picture is deterministic (no layout randomness). Smoke + determinism
  tests in `common/tests/test_plotting.py`.

## Phase 1

- Per the plan's "Review refinement 4", every STRONG opener marked KEEP was reworked (MERGE in
  practice) so the cell under `## Why this matters` actually carries the everyday picture and
  the listed gloss. A bare rename was never enough.
- Diagram scope held to the locked Q3 list (7 new in-notebook `def`s plus the shared helper).
  All other table "diagram opportunities" were deferred and noted by the section agents.

## Phase 2 (consistency read + fixes)

- Sections 04 (01-05) and 07 (01-04) agents had overwritten the notebook's H1 title cell with
  the `## Why this matters` block, dropping the `# Title`. Restored the original H1 above the
  heading in all nine, matching how 01/02/03/05/06 keep both in the first markdown cell.
- Removed three duplications where the new motivation block repeated the Intuition section
  almost verbatim: 04/03 (heights, 62/38), 04/05 (party map), 05/04 (parking a car). The image
  stays in Intuition; the motivation block now states the payoff plainly.
- 03/11: removed a "the fix is not a new algorithm" false-reframe (voice tell).
- Ruff F821 on the new diagram `def`s in 05/07 and 05/09: the string annotations referenced
  `matplotlib.axes.Axes` / `matplotlib.figure.Figure` but only `matplotlib.pyplot as plt` was
  imported. Added `import matplotlib` so the name binds (minimal, no annotation churn).
- `ruff format` collapsed two over-split signatures in 02/01 and 06/01.

### Out of scope, left as-is (pre-existing body prose, unchanged on main)

- 05/11 `rnn_and_lstm`: "beating the RNN by 80.8 to 87.3 points per seed, a mean margin of 83"
  sits in the results prose, not the motivation block, and came from the polish261114 run
  (PROGRESS records mean margin 83.1). Not touched.
- 06/05 `sentiment_classification` section 7: "a GRU ... trains about 20% faster here" where no
  GRU is built. Pre-existing claim in body prose. Flagged for a future voice pass, not changed
  in this motivation pass.

## Fix pass 1 (PR #4 review)

Addressed the Cursor, Codex and Copilot findings:

- BLOCKER 05/09 `cnn_mnist`: cell 0 had pasted `## 1. Intuition` / `## 2. Maths` /
  `## 3. From scratch` after the Why block, duplicating cell 2 verbatim. Stripped cell 0 back to
  H1 + Why. Also folded the pre-Why "An MLP treats an image as 784 unrelated numbers" opener into
  one sentence of Why (the shuffle-the-pixels point), so Why is the single motivation.
- SHOULD-FIX dedup (Why repeated the Intuition image near word-for-word): 04/01 `kmeans` (flags in
  a field), 03/22 `svm` (widest road / support vectors), 03/26 `scorecard` (Age 25-34 earns 12,
  utilisation over 80% costs 30). Image/example kept in Intuition; Why reworked to payoff and knobs.
- NIT 05/03 `optimisers`: Why used a shopping-trolley valley, Intuition a ball in a valley. Gave
  Why a distinct picture (a trolley with one stiff wheel that veers), leaving the valley to Intuition.
- Codex 02/02 `bayes`: qualified "a 1% base rate keeps the answer low no matter how good the test".
  Now ties the low answer to the example's 9% false-positive rate (answer near 9%), and notes a far
  lower false-positive rate would let a positive overcome the small prior. Numbers are the notebook's.
- NIT 06 em dashes in code: removed from print strings / titles / one comment in 04, 05, 06. En
  dashes in numeric ranges (150-200, 1-2, 83-85) left as-is per the house rule the reviewer agreed with.
- Copilot `common/plotting.py::draw_computation_graph`: added an up-front `ValueError` when `nodes`
  is empty (previously `min()`/`max()` crashed on an empty `nodes` with empty `edges`), plus a test.

### Pushed back

- NIT 05/09 `plot_filter_grid` returns a `Figure` with no optional `ax`. Left as-is. It draws a grid
  of subplots, so a single caller-supplied `ax` has no meaning, and the plan's review refinement 3
  explicitly allows returning the `Figure`. The reviewer agreed it is acceptable.
