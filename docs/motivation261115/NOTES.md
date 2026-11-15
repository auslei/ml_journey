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
