# Progress and handoff

Last updated: 2026-11-13. Repo: https://github.com/auslei/ml_journey (public). Sections 01 to 05 are merged into `main` and pushed. Section 06 is complete on branch `section-06-nlp`, pending merge.

## Goal

One project that walks from linear algebra to LLM fine-tuning, as a personal refresher and blog training material. Read `CLAUDE.md` first. It holds the standards, the notebook template and the writing style (`~/dev/my_voice.md`).

## Status

| Section | State | Notes |
|---|---|---|
| 01 linear algebra and calculus | done | 5 notebooks, autodiff from scratch |
| 02 probability | done | basics, Bayes, MLE/MAP bridge to ML |
| 03 classical ML | done | regression (part A) and classification, trees, ensembles, metrics, scorecard (part B) |
| 04 unsupervised | done | k-means, hierarchical, DBSCAN, GMM, PCA, t-SNE, UMAP, evaluation |
| 05 deep learning | done | 12 notebooks, intuition test, README, TensorFlow appendix in its own env (see decisions) |
| 06 NLP | done | 6 notebooks + intuition test; nltk, gensim, spacy added; native PyTorch throughout |
| 07 LLM | not started | |

Full suite: section 06 suite 7 passed, ruff clean. Full suite 101/103 passed (2 pre-existing MPS failures in section 05: `08_pytorch_basics` float64/MPS error, `10_transfer_learning` timeout). Last commit: `b9d2f7e` (section-06-nlp branch). The TF appendix passes separately in `.venv-tf` (1 test).

## Decisions made

- Keras 3 is the teaching API (runs on the PyTorch backend). Native PyTorch where the training loop is the lesson, and for NLP and LLM. TensorFlow-only material (`tf.data`, `GradientTape`) goes in an appendix at the end of section 05. Confirm Keras 3 backend support in the installed version before pinning.
- TensorFlow cannot share an environment with `umap` (numba and TF load conflicting LLVM copies, so `import umap` aborts the kernel once TF is installed). The appendix runs in a second env: `UV_PROJECT_ENVIRONMENT=.venv-tf uv run --extra torch --extra tf pytest 05_deep_learning/appendix`. `05_deep_learning/conftest.py` skips it when TF is absent.
- GPU: `common.dl.get_device()` picks cuda, then mps, then cpu. Prose numbers in 05 were read off CUDA runs and shift by a few points on CPU.
- Python 3.12, `uv`, one `pyproject.toml`. `umap-learn` is a core dependency. Optional extras: `torch`, `tf`, `llm`, `nlp`.
- `common/` is an installed package. Use `from common.datasets import ...` and `from common.plotting import ...`. Datasets come from sklearn built-ins, seeded generators, or download-and-cache into the gitignored `data/` folder.
- Notebooks are committed with outputs stripped (`nbstripout`). Each stays under 30KB.
- Prose depth is one markdown cell per code cell, about 500 words per notebook. The user said this is fine for now. A deeper blog-quality pass on 01 and 02 can wait for the end.
- Source folders under `/home/anthony/github_backup/` are read-only. Their `.git` directories were removed earlier at the user's request.

## Next steps

1. **Merge section 06 to main and push** — ask the user to confirm before doing so.
2. **Section 07 LLM.** Reuse `llm_finetune` (`src`, `scripts`, `tests`, notebooks) at `/Volumes/SSD_SAM/github_backup/llm_finetune` and the HuggingFace notebooks `h0` to `h2` from `machine_learning_deep_learning`. Add a concept notebook on tokenisation and LoRA. Do not copy model weights or data (402MB). Add the `llm` extra only when needed; check it does not clash with `umap` or `torch`.
3. Root `README.md`: roadmap and setup, written last.
4. Deferred prose pass: 05 notebooks run 700 to 900 words against the 500 target, and 01 and 02 need the blog-quality pass. Notebook 11 (RNN and LSTM) has a seed-dependent result where the LSTM does not beat the vanilla RNN on the delayed-recall task. The prose reports it honestly, but a better task or training budget could be tried.
5. Integration check, a voice read-through of all prose, then ask the user whether to merge to main and push. Sections 01 to 05 are on `main` (the user approved the merge and a public push on 2026-11-11). Ask again before each later merge or push.

## Lessons for the next run

- The coordinator re-runs `uv run pytest` and ruff after every sub-agent, then commits. Sub-agent reports have matched so far, but check anyway.
- The full suite takes about 6 minutes, so run it in the background. First-time dataset downloads are slow (CIFAR-10 took about 15 minutes once). They are cached in `data/` now, but a fresh clone will hit them again.
- `nbconvert` is not installed. To see notebook outputs, run the notebook with `nbclient.NotebookClient` or just run `pytest` on it.
- Do not import torch and TensorFlow-dependent libraries carelessly. Anything that needs TensorFlow goes in the `.venv-tf` environment only.
- Prose numbers must come from a real run. Notebooks are stored stripped, so quote numbers only after executing them. Hedge CPU/GPU differences.

## How to resume

```
cd /Users/auslei/dev/anthonysun.ai/ml_journey
git checkout main && git pull
git checkout -b section-07-llm
uv sync --extra torch --extra nlp
uv run --extra torch --extra nlp pytest 06_nlp/
```

Source backup is at `/Volumes/SSD_SAM/github_backup/` (external SSD — must be mounted).

Optional, only to run the TensorFlow appendix:

```
UV_PROJECT_ENVIRONMENT=.venv-tf uv sync --extra torch --extra tf
UV_PROJECT_ENVIRONMENT=.venv-tf uv run --extra torch --extra tf pytest 05_deep_learning/appendix
```

Then ask Claude to read `CLAUDE.md` and this file, and plan section 07. Workflow in use: plan, confirm, sub-agents per notebook run sequentially or in parallel where independent, each owns its tests, coordinator re-runs the gates before committing.
