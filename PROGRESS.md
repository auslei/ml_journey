# Progress and handoff

Last updated: 2026-11-10. Branch: `consolidate-ml-journey` (not merged, not pushed).

## Goal

One project that walks from linear algebra to LLM fine-tuning, as a personal refresher and blog training material. Read `CLAUDE.md` first. It holds the standards, the notebook template and the writing style (`~/dev/my_voice.md`).

## Status

| Section | State | Notes |
|---|---|---|
| 01 linear algebra and calculus | done | 5 notebooks, autodiff from scratch |
| 02 probability | done | basics, Bayes, MLE/MAP bridge to ML |
| 03 classical ML | done | regression (part A) and classification, trees, ensembles, metrics, scorecard (part B) |
| 04 unsupervised | done | k-means, hierarchical, DBSCAN, GMM, PCA, t-SNE, UMAP, evaluation |
| 05 deep learning | not started | |
| 06 NLP | not started | |
| 07 LLM | not started | |

Full suite: `uv run pytest` (55 passed, about 2 minutes), `uv run ruff check .` and `uv run ruff format --check .` clean.

## Decisions made

- Keras 3 is the teaching API (runs on the PyTorch backend). Native PyTorch where the training loop is the lesson, and for NLP and LLM. TensorFlow-only material (`tf.data`, `GradientTape`) goes in an appendix at the end of section 05. Confirm Keras 3 backend support in the installed version before pinning.
- Python 3.12, `uv`, one `pyproject.toml`. `umap-learn` is a core dependency. Optional extras: `torch`, `tf`, `llm`.
- `common/` is an installed package. Use `from common.datasets import ...` and `from common.plotting import ...`. Datasets come from sklearn built-ins, seeded generators, or download-and-cache into the gitignored `data/` folder.
- Notebooks are committed with outputs stripped (`nbstripout`). Each stays under 30KB.
- Prose depth is one markdown cell per code cell, about 500 words per notebook. The user said this is fine for now. A deeper blog-quality pass on 01 and 02 can wait for the end.
- Source folders under `/home/anthony/github_backup/` are read-only. Their `.git` directories were removed earlier at the user's request.

## Next steps

1. Section 05 deep learning. Plan it as parts, one sub-agent each, run in order:
   - 05a: NN basics, backprop, optimisers, learning rate schedules, regularisation, unstable gradients, hyperparameter tuning. Reuse `machine_learning_deep_learning` notebooks 0, 1, 1.1, 1.2, 2, 3, 4, 6, 7, 8, `9.1_auto_diff`.
   - 05b: CNN (MNIST, plus catsdogs, transfer learning from `5_transfer_learning`), RNN and LSTM (C1 to C4, `a2.pytorch_rnn`), a short PyTorch basics notebook (`a1`), optional VAE.
   - 05 appendix: TensorFlow-specific material.
   - Add MNIST and CIFAR-style loaders to `common/datasets.py`. Add the `torch` extra (and Keras 3) to the environment.
2. Section 06 NLP. Reuse `Word-Embedding` (CBOW, skip-gram), `python_gensim_nlp` (LDA, Word2Vec), `auto_regressive_model`, and `python_topic_modelling`. Add a short attention and transformer notebook.
3. Section 07 LLM. Reuse `llm_finetune` (`src`, `scripts`, `tests`, notebooks) and the HuggingFace notebooks `h0` to `h2`. Add a concept notebook on tokenisation and LoRA. Do not copy model weights or data (402MB).
4. Root `README.md`: roadmap and setup, written last.
5. Integration check, a voice read-through of all prose, then ask the user whether to merge to main and push.

## How to resume

```
cd /home/anthony/github_backup/ml_journey
git checkout consolidate-ml-journey
uv sync
uv run pytest 01_linear_algebra_calculus
```

Then ask Claude to read `CLAUDE.md` and this file, and continue with section 05. Workflow in use: one sub-agent per part, run sequentially, each owns its tests, and the coordinator re-runs the gates before committing.
