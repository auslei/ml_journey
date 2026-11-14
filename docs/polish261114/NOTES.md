# polish261114 — build notes

Minor assumptions and decisions made during the build. Major concerns go to QUESTIONS.md
under "Build-time questions" and stop the run.

## Assumptions

- **Item 3 task choice (Q3): adding problem, not noisy delayed recall.** Noisy recall was tried
  first (per Q3 default). It does not give a clean LSTM win: on a CPU 5-seed sweep the two models
  either both saturate at 100% (ties, so "every seed" fails) or the LSTM fails to optimise at
  longer lengths and scores below the RNN. No length/epoch window gave LSTM > RNN on every seed.
  Switched to the plan's named fallback, the adding problem (Hochreiter and Schmidhuber 1997). On
  a CPU 5-seed sweep the LSTM beats the vanilla RNN on every seed at lengths 20/30/40 with mean
  margins of 70 to 85 points, far above the 15-point Q4 bar. Chosen config: length 30, 30 epochs,
  5 seeds. Both the noisy-recall generator and the adding-problem generator are added to
  `common/sequences.py`; the noisy-recall one is kept (it is a valid seeded task with a test) but
  the notebook trains on the adding problem.
- **Determinism (Q4): trained fits forced to CPU** in notebook 11, per the section 07 precedent,
  so the quoted numbers reproduce under nbmake on any machine (this Mac resolves `get_device()` to
  mps). The gradient-at-init measurements are also computed on CPU for the same reason.
- The adding problem is regression. "Accuracy" is the fraction of test items with absolute error
  below 0.04 (the classic adding-problem success threshold). The RNN sits near 0.15, which is the
  predict-the-mean baseline.
- **nb11 length: lands at ~810 words (naive count), down from 899 baseline.** The two display
  equations and inline LaTeX in the maths section inflate the whitespace count and set a floor
  well above the 500 target. The maths is the teaching core of this notebook, so it was tightened
  rather than cut. Per the plan's risk note, nb11 is the item-3 notebook, not an item-2 trim.
- Q4 result: at length 30, 30 epochs, CPU, seeds 0-4, LSTM beats the vanilla RNN on every seed
  (margins 80.8/81.0/87.3/82.8/83.7, mean 83.1), GRU solves it at 1.00. Bar cleared with 5 seeds.
