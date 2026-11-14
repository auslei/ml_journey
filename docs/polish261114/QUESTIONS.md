# polish261114 — open questions

Answer before building. Each has a default so work can start if you say nothing.

## Q1. Word ceiling for the blog-quality pass on 01 and 02

01 and 02 notebooks already sit at 441 to 631 words, at or below the 500 target. "Blog-quality"
usually means more intuition, which adds words. This pulls against item 2 (trim 05 to 500).

**Default:** let 01 and 02 grow where depth helps, up to roughly 700 to 900 words, capped by the
30KB stripped-notebook limit. 05 still trims toward 500. The two sections keep different word
rules.

## Q2. Does the 500-word target apply to section 05 `intuition_test.ipynb`?

It runs 1313 words but is a different format (questions, predictions, answers) that is naturally
longer. The other intuition tests (01 at 631, 02 at 441) are not uniform either.

**Default:** intuition tests are exempt from the 500 target. Trim only for filler, not to hit a
number.

## Q3. Notebook 11 — which replacement task?

Candidates: noisy delayed recall (random symbols fill the gap instead of a blank, smallest
change to the existing helper) or the adding problem (sum two marked values).

**Default:** try noisy delayed recall first; fall back to the adding problem only if the LSTM
win is not reliable. It is acceptable to raise sequence length and epochs to force a clear gap.

## Q4. What counts as the LSTM "reliably beating" the vanilla RNN?

Need a pass threshold before tuning.

**Default:** CPU deterministic run, 5 seeds, LSTM mean test accuracy above the RNN on every
seed with a mean margin of at least 15 points at the chosen length. If only 3 seeds are
practical given run time, 3 is acceptable.

## Q5. Disposition of the root README "Future plans" section

Its three items are exactly what this feature delivers, so they will be done.

**Default:** remove the three completed items. If there is genuinely remaining future work,
tell me and I will keep a short honest line; otherwise drop the section.

## Note (not blocking)

Merge and push of `polish261114` to `main` stays gated on your explicit approval after item 6
passes, per the standing workflow rule. No action needed now.
