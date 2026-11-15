# Plan: motivation261115

Feature: add a short, plain-English "Why this matters" motivation at the top of each
content notebook, a gentle gloss on the complex calculation or key parameters where one
is needed, a small number of genuinely missing diagrams (written as functions, not inline
spaghetti), and better comments on sample-data generation code.

Branch: `motivation261115` (stay on it). Plan only. No implementation until approved.

## The honest headline

I surveyed the opening cells, sample-data code and diagram coverage of all 47 content
notebooks (sections 01 to 07). Two facts shape the whole plan:

1. **Most openers already motivate well.** Sections 01 and 02 got the blog-quality pass in
   `polish261114`; 03 to 05 open with vivid everyday pictures (XOR, a ball in a valley, a
   message down a line of people, flags in a field, a shadow on a wall); 06 and 07 open with
   concrete bottlenecks and token counts. Of 47 notebooks, **43 already have a STRONG
   motivation opener, 4 are PARTIAL/thin, 0 are absent.** So the dominant action is **merge
   and standardise, not add**. Writing a second motivation block on top of an existing one
   would duplicate content, which `CLAUDE.md` forbids.

2. **The real gaps are the two things Anthony flagged last:** sample-data comments ("I find
   the comment is lacking when generating sample data") and parameter glosses. Those are
   consistent across every section and are where most of the value is.

So this plan deliberately spends little effort on notebooks that already read well, and
concentrates on: (a) a consistent motivation *heading* so every notebook visibly has the
section, absorbing existing prose rather than repeating it; (b) a parameter/calculation
gloss only where a beginner would stall; (c) sample-data comments everywhere they are thin;
(d) a short list of diagrams that are genuinely missing, each written as a function.

## Scope

In: the 47 numbered content notebooks across 01 to 07.

Out (default, see QUESTIONS Q5):
- `intuition_test.ipynb` and `regression_intuition_test.ipynb` in every section. These are
  the closing exercises (predict-then-run); a motivation banner would spoil them.
- `05_deep_learning/appendix/tensorflow_data_and_gradient_tape.ipynb`. It is an appendix in
  a second environment (`.venv-tf`), off the main path, and runs nbmake only when TF is
  installed.

## Conventions this feature introduces

### Motivation block

- A single markdown cell at the very top of each notebook, headed `## Why this matters`.
- 120 to 200 words. Short enough to read in under a minute, in Anthony's voice (short
  sentences, plain words, Australian spelling, no em dashes, numbers only from real runs).
- Content, in this order: one line on why the topic matters in practice; a plain-English
  picture or everyday example; where the topic has a hard calculation or a dial a beginner
  will fiddle with, one or two sentences naming the knob, what it does, and a typical value.
- **Counted inside the existing ~500-word notebook target, not on top of it** (see QUESTIONS
  Q2). For STRONG notebooks the existing first markdown cell already is this content, so the
  action is to retitle/trim it to the convention, adding little or no net words. For PARTIAL
  notebooks it is net new prose, still folded into the 500.
- Duplication rule: if the notebook already opens with motivation (STRONG), do **not** add a
  second block. Adapt the existing cell in place. The table below marks each notebook KEEP
  (retitle only, no new content), MERGE (light rework of existing opener), or ADD (new prose
  because the opener is thin).

### Parameter / calculation gloss

- Only where the table marks "gloss". A gloss is one to three sentences, either inside the
  motivation block (if short) or as a short note next to the maths cell it explains.
- Pattern: name the knob, say what moving it does, give a typical value or range. Example for
  SVM: "`C` trades margin width against mistakes. Small `C` (0.01) gives a wide margin that
  tolerates errors; large `C` (100) chases every point. Typical search: 0.01 to 100 on a log
  scale."
- No new derivations. The maths sections already carry those; the gloss is the plain-English
  companion, not a second derivation.

### Diagrams

- Add a diagram only where a complex idea currently has **no** visual and the picture
  materially helps. The notebooks already plot heavily, so this is a short list, not a sweep.
- Every new diagram is a **function with a docstring and type hints**, never inline
  procedural plotting. Matplotlib only. No new dependencies.
- Placement rule: a plot used by **one** notebook is a `def` with a docstring
  defined in that notebook. A plot used by **two or more** notebooks goes in
  `common/plotting.py` with a unit test. Do not create `common/diagrams.py` (see QUESTIONS
  Q4); `common/plotting.py` is small (82 lines) and the right home.
- The only cross-notebook candidate is a computation-graph schematic wanted by both
  `01/04_autodiff` and `05/02_backprop`. If both are built, add one
  `draw_computation_graph(...)` helper to `common/plotting.py` with a test. Everything else
  stays in-notebook.

### Sample-data comments (the main ask)

- Standard: every cell or block that generates or synthesises sample data gets a short
  comment block (two to four lines) that states, in intent terms not line-by-line:
  - what it produces (the name, and what each column / channel / class means),
  - the shape and dtype of the result,
  - what the key numbers mean and why they were chosen for the lesson (noise level, class
    separation, counts, number of layers),
  - why the seed is fixed (same figures on every run, so the prose numbers stay true).
- One comment per logical block, not per line. Comments explain intent, never restate code.
- `common/datasets.py` and `common/sequences.py` are **already compliant**: every generator
  has a docstring covering shapes, parameters and the "why". The only module gap is
  `07_llm/llm_finetune/model.py::_corpus_text` (extraction logic unexplained). The bulk of
  the work is the in-notebook sample-data cells listed per section below.

## Standards that still apply (checklist for every touched notebook)

- Notebook order unchanged (intuition, maths, cost, from-scratch, library, tuning, breaks).
  The motivation block sits above the existing "Intuition" section.
- Every random call stays seeded. Outputs stripped with `nbstripout` before commit.
- Each notebook stays under 30KB after stripping. Current max is 24KB
  (`13_bias_variance_regularisation`); a 120 to 200 word block adds ~1 to 1.5KB, so headroom
  is fine, but the consistency pass re-checks sizes.
- Australian spelling, no em dashes, no self-answered rhetorical questions, no "it's not X
  it's Y" reframes. Numbers in prose come from a real run only.
- Every touched notebook passes `pytest --nbmake` (with the right extras per section).
- `ruff check` and `ruff format --check` clean on any touched `.py`.

## Per-notebook plan

Opener = current state (STRONG / PARTIAL). Action = KEEP (retitle existing opener to the
convention, ~no new words) / MERGE (light rework) / ADD (new prose). Gloss = complex
calc/parameter that needs a plain-English line. Diagram = new picture needed (inline unless
marked shared); "has" = already plotted, nothing to add. Data comments = sample-data cells
whose comments are thin.

### 01 linear algebra and calculus

| Notebook | Opener | Action | Gloss | Diagram | Data comments |
|---|---|---|---|---|---|
| 01_vectors_and_matrices | STRONG | KEEP | determinant sign/collapse, condition number | has (transforms) | rng seed cell; random matrix shapes; `transform_demo` shapes |
| 02_decompositions | STRONG | KEEP | eigen/SVD read, normal-equation κ², ridge | inline: low-rank image at ranks k; variance bar | why symmetric 2x2; why trig "image"; `1e-6` collinearity scale |
| 03_calculus_and_gradients | STRONG | KEEP | LR stability `2/λmax`, finite-diff step ~1e-5 | inline: loss trajectory per LR (divergence) | `f`=cubic why; `surface` bowl why; meshgrid range; LR sweep points |
| 04_autodiff | STRONG | KEEP | adjoint accumulation (add, don't overwrite), vanishing | shared: computation-graph schematic | line fit `2+3x`, noise 0.1, n=50 why; `messy` fn why hard |

### 02 probability

| Notebook | Opener | Action | Gloss | Diagram | Data comments |
|---|---|---|---|---|---|
| 01_probability_basics | STRONG | KEEP | PDF vs PMF (density can exceed 1), CLT `σ/√n` | inline: running-mean LLN convergence | CLT n=1,5,50 choice; contingency 0.30/0.10 rates |
| 02_bayes | STRONG | KEEP | base-rate, Beta(a,b) as pseudo-counts, NB var smoothing | inline: prior vs flat convergence vs n | base-rate rates; `var_smoothing` `.max()` |
| 03_probability_in_ml | STRONG | KEEP | MLE↔loss, `λ=σ²/τ²`, softmax max-subtract | inline: logistic curve or separable-loss blow-up | BCE clip `1e-12` why; `τ` grid 0.1..100 why |

### 03 classical ML

| Notebook | Opener | Action | Gloss | Diagram | Data comments |
|---|---|---|---|---|---|
| 00_ml_workflow | STRONG | KEEP | CV folds, leakage via preprocessing | has | housing n=4000 (already good) |
| 10_linear_regression | STRONG | KEEP | normal equation, scaling/condition number | inline: cost surface zigzag, raw vs scaled | bias column (good); split seed note |
| 11_polynomial_and_features | STRONG | KEEP | column blow-up `C(p+d,d)`, raw-power conditioning | has | `make_noisy_curve` (good), add one line on noise 0.3 |
| 12_gradient_descent_variants | STRONG | KEEP | `η<2/λmax`, momentum, LR decay, patience | has | exact-solution baseline comment |
| 13_bias_variance_regularisation | STRONG | KEEP | soft-threshold, L1 zeroing, `α`↔`C` | has | `make_sparse_regression` (good) |
| 20_logistic_regression | STRONG | KEEP | sigmoid deriv, log-loss convex, `C`, separable blow-up | inline: log-loss penalty curve | iris 2-species subset note |
| 21_knn | STRONG | KEEP | `k` as bias/variance dial, Minkowski, curse | has | argpartition trick (good) |
| 22_svm | STRONG | KEEP | hinge, `C`, RBF `gamma` (small=smooth, large=local) | has | subgradient data note |
| 23_decision_trees | STRONG | KEEP | Gini/entropy, max_depth, pruning | has | split-search data note |
| 24_ensembles | STRONG | KEEP | variance-of-average, OOB, LR x n_trees | has | bootstrap/residual-fit notes |
| 25_classification_metrics | STRONG | KEEP | F1 harmonic, AUC rank meaning, `t*=cFP/(cFP+cFN)` | has | `roc_scratch` sweep note |
| 26_scorecard_capstone | PARTIAL | ADD | WoE, IV, PDO, KS, PSI | has | 0.5 smoothing (good) |

### 04 unsupervised

| Notebook | Opener | Action | Gloss | Diagram | Data comments |
|---|---|---|---|---|---|
| 01_kmeans | STRONG | KEEP | `k` choice, k-means++, scaling, round-cluster assumption | has (rich) | "what this dataset teaches" on blobs/customers/moons/unequal |
| 02_hierarchical_and_dbscan | STRONG | KEEP | linkage choice, DBSCAN `eps`/`min_samples` | has | bridge construction; tight+loose blobs |
| 03_gaussian_mixtures | STRONG | KEEP | covariance types, BIC, `reg_covar`, EM | has | 1D mixture specs; `make_tilted` specs why |
| 04_pca | STRONG | KEEP | variance≠relevance, scaling, whitening, `k` by variance | has | why scaling flips dominance (wine) |
| 05_tsne_and_umap | STRONG | KEEP | perplexity, `n_neighbors`/`min_dist`, trustworthiness caveat | has | 10D three-Gaussian spreads; noise clouds |
| 06_evaluating_unsupervised | PARTIAL | ADD | silhouette/DB/CH circularity, stability ARI, downstream CV | has | customer/moons "teaches" note |

### 05 deep learning

| Notebook | Opener | Action | Gloss | Diagram | Data comments |
|---|---|---|---|---|---|
| 01_neural_net_basics | STRONG | KEEP | activation choice, standardisation effect | has | XOR array; moons/mnist/housing seed note |
| 02_backprop_and_autodiff | STRONG | KEEP | tanh deriv `1-h²`, grad-check | shared: computation-graph schematic | why 6 samples, random weights seed |
| 03_optimisers | STRONG | KEEP | betas (0.9/0.999), bias correction, LR scale per optimiser | has | Rosenbrock A=10 why; LR choices |
| 04_learning_rate_schedules | STRONG | KEEP | warmup fraction, cosine, one-cycle | has | schedule constants note |
| 05_unstable_gradients | STRONG | KEEP | sigmoid slope 0.25, He vs Glorot, clip | has | why 30 layers / 4 scales |
| 06_regularisation_and_callbacks | STRONG | KEEP | L2 `(1-ηλ)`, dropout `p` & `1/(1-p)`, patience | has | why 300 training rows (exaggerate overfit) |
| 07_hyperparameter_tuning | PARTIAL | ADD | log-uniform LR, winner's curse | inline: trial scatter / (depth,width) heatmap | `10**uniform(-4,-1)` why log |
| 08_pytorch_basics | STRONG | KEEP | requires_grad, zero_grad, train/eval, dtype | has | single-weight example (good) |
| 09_cnn_mnist | PARTIAL* | MERGE | param count, pooling, shift equivariance | inline: learned-filter grid | why vertical filter `[[-1,0,1]]` |
| 10_transfer_learning | STRONG | KEEP | frozen vs fine-tune, BN eval mode, LR 1e-4 | inline: frozen-feature 2D scatter (optional) | cats/dogs filter; ImageNet norm (good) |
| 11_rnn_and_lstm | STRONG | KEEP | grad product over time, LSTM additive cell, clip | has | memory/adding task; Wh singular-value 0.5/1/1.5 why |
| 12_vae | STRONG | KEEP | reparam trick, KL+recon, `β`, posterior collapse | has | why `β` grid 0..100 |

\* 09 is borderline STRONG (strong shuffle experiment); MERGE = tighten the opener, add the
param-count line, no full rewrite.

### 06 NLP

Openers all STRONG, but note: section 06 notebook prose did not get the voice read-through
(only its READMEs did), so the consistency pass should check voice here more closely.

| Notebook | Opener | Action | Gloss | Diagram | Data comments |
|---|---|---|---|---|---|
| 01_text_preprocessing | STRONG | KEEP | TF-IDF `log(N/df)`, sparsity | inline: token count before/after stopwords (none yet) | Austen excerpt shape; toy BoW corpus |
| 02_word_embeddings | STRONG | KEEP | window=2, emb dim trade-off, negative sampling | has (t-SNE) | CBOW pairs; why vector_size=50 |
| 03_topic_models | STRONG | KEEP | Dirichlet `α` sparsity, `K` by coherence | has (coherence) | toy 30-doc 3-theme corpus labels |
| 04_sequence_models_text | STRONG | KEEP | RNN recurrence, perplexity, clip, SEQ_LEN, temperature | has (loss) | 8K-char slice; batch shift-by-1 shapes |
| 05_sentiment_classification | STRONG | KEEP | packed sequences, freeze-then-finetune, dropout 0.3 | inline: BoW vs LSTM accuracy bar (none yet) | movie_reviews balance; vocab-from-train; pad/OOV |
| 06_attention_transformer | STRONG | KEEP | `QK^T/√d_k` scaling, heads, positional sinusoids, `d_ff` | has (PE + attn heatmaps) | toy Q/K/V; toy sentence vocab |

### 07 LLM

| Notebook | Opener | Action | Gloss | Diagram | Data comments |
|---|---|---|---|---|---|
| 01_tokenisation | STRONG | KEEP | BPE merges, deterministic tie-break, vocab vs seq-length | inline: pair-frequency or vocab-vs-length (optional) | toy corpus `low/lower/newest/widest` why |
| 02_huggingface_basics | STRONG | KEEP | tokenizer-model pairing, hidden-state shape | has | why DistilBERT ("small, fast") one line |
| 03_lora_from_scratch | STRONG | KEEP | `r`, `alpha` (`alpha/r` scale), `B=0` init, target modules | inline: A/B matrix schematic (optional, candidate shared) | LoRALinear 64x64 (good) |
| 04_finetuning_walkthrough | STRONG | KEEP | TrainConfig knobs, perplexity same-tokenizer only | inline: loss curve / before-after perplexity bar | `demo_corpus` (good); Qwen gate (good) |

Plus one module edit: `07_llm/llm_finetune/model.py::_corpus_text` gets the sample-data
comment standard (why instruction/input/output are extracted, what "seed the tokenizer"
means).

## Build order (sub-agent friendly), with gates

Sections are independent of each other and talk only through `common/`, so the per-section
passes parallelise. The one cross-cutting dependency is the shared computation-graph helper,
which must land before 01 and 05 use it.

**Phase 0: shared plotting helper (if approved).** One agent adds
`draw_computation_graph(...)` to `common/plotting.py` and a unit test in `common/tests/`
(renders without error, returns an `Axes`, deterministic node positions for a fixed input).
Gate: `uv run pytest common/` green, `ruff check common/` clean. ~10 min. Skip this phase
entirely if QUESTIONS Q3 says keep all diagrams in-notebook.

**Phase 1: per-section notebook passes.** One sub-agent per section (seven agents), run in
parallel where CPU allows; 05 is the long pole. Each agent, for its section only:
1. applies the motivation convention (KEEP/MERGE/ADD per the table),
2. adds the parameter glosses listed,
3. adds the listed diagrams as functions (in-notebook, or calling the shared helper),
4. brings the listed sample-data cells up to the comment standard,
5. strips outputs, then runs `pytest --nbmake` on its own notebooks plus `ruff` on any `.py`
   it touched, and reports word counts and post-strip sizes per notebook.

Contract each agent is handed: the convention block above, its section's table row set, the
helper signature if used, and nothing from other sections. Gate per section: that section's
nbmake green, ruff clean, every notebook under 30KB, motivation 120 to 200 words.

Extras per section for nbmake: 01/02/03/04 need only `uv run pytest`; 05 needs
`--extra torch`; 06 needs `--extra torch --extra nlp` (plus the spaCy model and NLTK data
already cached); 07 needs `--extra torch --extra llm` (offline tiny GPT-2).

**Phase 2: consistency read.** One agent (coordinator) reads every touched notebook top to
bottom for: voice (especially 06, whose prose was never voice-passed), no duplicated
motivation, Australian spelling, no em dashes, no numbers that did not come from a real run,
sizes under 30KB, and that each notebook visibly carries the `## Why this matters` heading.
Then run the full suite: `uv run --extra torch --extra nlp --extra llm pytest` and both ruff
checks. Resolve any failure before declaring done. Gate: full suite green.

**Phase 3: cleanup + commit.** Update section READMEs only if a diagram or section changed
what a reader should be able to explain (most will not change). Strip outputs, commit per
section or as one reviewed change, then ask Anthony about merge. Never merge automatically.

### Run-time estimate

- Phase 0: ~10 min (one small helper + test).
- Phase 1: per-section nbmake wall times roughly 01 ~40s, 02 ~30s, 03 ~90s, 04 ~120s (umap),
  05 ~240s (torch, CIFAR cached), 06 ~150s, 07 ~120s. With editing and ruff, each agent is
  ~15 to 30 min; parallel, Phase 1 wall is ~30 to 45 min, bounded by 05 and CPU contention.
- Phase 2: full nbmake ~6 min plus a careful read ~15 to 20 min.
- Phase 3: ~10 min.

Total: roughly 1 to 1.5 hours of agent wall time, most of it in the per-section passes and
the final read. First-run dataset downloads are already cached, so nbmake should not hit the
slow CIFAR path again.

## What this plan deliberately does not do

- It does not add a motivation block on top of the 43 STRONG notebooks. They are retitled to
  the convention, not rewritten. This is the `CLAUDE.md` "do not duplicate" rule in force.
- It does not create `common/diagrams.py` or a diagram wrapper around `ax.bar` / `ax.plot`.
  One-line plots stay one line; only genuinely reusable, non-trivial pictures become helpers.
- It does not touch intuition tests or the TF appendix (QUESTIONS Q5), nor re-document
  `common/datasets.py` / `common/sequences.py`, which already meet the comment standard.


## Review refinements

Added by the engineering-manager review on 15 Nov 2026. Where these conflict with anything above, these win.

1. **Counts corrected.** The tables list 43 STRONG and 4 PARTIAL (26_scorecard_capstone, 06_evaluating_unsupervised, 07_hyperparameter_tuning, 09_cnn_mnist), not 41 and 6. I fixed the text above and in QUESTIONS.md to match.
2. **Diagram scope locked to the Q3 list.** The tables propose more inline diagrams than QUESTIONS Q3 recommends. Unless Anthony picks the ambitious option, build only: the shared computation-graph schematic (01/04, 05/02), loss trajectory per LR (01/03), running-mean convergence (02/01), learned-filter grid (05/09), trial scatter (05/07), the first plots for 06/01 and 06/05, and the loss or perplexity bar for 07/04. Treat the other table entries (01/02 low-rank and variance bar, 02/02 prior convergence, 02/03 logistic curve, 03/10 cost surface, 03/20 log-loss curve, and everything marked optional) as deferred. Note them in the build report and don't build them.
3. **Every new diagram is a tested function.** In-notebook plotting `def`s take data and an optional `ax`, return the `Axes` (or `Figure`), and use no randomness of their own (pass seeded data in). They are smoke-tested by nbmake. If a helper is reused, or runs past about 20 lines, move it to `common/plotting.py` with a test in `common/tests/test_plotting.py` (Agg backend, renders without error, returns an `Axes`, same output on a repeat call with the same input).
4. **Motivation content check.** Retitling a STRONG opener only counts as done if the cell under `## Why this matters` actually covers why it matters and gives an everyday picture, and includes the gloss where the table lists one (or links to the note beside the maths cell). If it doesn't, change the action to MERGE. Don't just rename the heading.
5. **Comment standard also covers in-notebook generator functions** and the seed line itself (one short line saying why it is fixed). Don't add any line-by-line comments.
6. **Workflow tail.** After Phase 2, Claude Code opens a PR from `motivation261115`, then a Cursor review runs. Merge waits for Anthony. After merge, move `docs/motivation261115` to `docs/completed/motivation261115` and push to main.
7. **Estimate widened.** Seven parallel agents on one Mac will fight for CPU during nbmake, so allow about 1.5 to 2.5 hours of wall time for the build and final read, plus PR and review time.
