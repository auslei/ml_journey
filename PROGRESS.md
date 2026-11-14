# Progress and handoff

Last updated: 2026-11-15. Repo: https://github.com/auslei/ml_journey (public). Sections 01 to 07 are merged into `main`. The `polish261114` branch holds the final polish pass (prose depth and length, the notebook 11 rewrite, voice read-through), pending review.

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
| 07 LLM | done | 4 notebooks + intuition test; `llm_finetune` package on transformers + peft + Trainer; tiny GPT-2 offline, Qwen opt-in |

Section 07 gates (branch `llm07261114`): `pytest 07_llm/ common/` 99 passed offline (41 unit tests, 5 notebooks, common suite), ruff check and ruff format clean, the four content notebooks plus the intuition test pass `--nbmake` offline. The `llm` extra adds transformers 5.22.2, peft 0.21.2, datasets and accelerate, and is independent of the `nlp` extra; `import umap` still works alongside it.

## Decisions made

- Keras 3 is the teaching API (runs on the PyTorch backend). Native PyTorch where the training loop is the lesson, and for NLP and LLM. TensorFlow-only material (`tf.data`, `GradientTape`) goes in an appendix at the end of section 05. Confirm Keras 3 backend support in the installed version before pinning.
- TensorFlow cannot share an environment with `umap` (numba and TF load conflicting LLVM copies, so `import umap` aborts the kernel once TF is installed). The appendix runs in a second env: `UV_PROJECT_ENVIRONMENT=.venv-tf uv run --extra torch --extra tf pytest 05_deep_learning/appendix`. `05_deep_learning/conftest.py` skips it when TF is absent.
- GPU: `common.dl.get_device()` picks cuda, then mps, then cpu. Prose numbers in 05 were read off CUDA runs and shift by a few points on CPU.
- Python 3.12, `uv`, one `pyproject.toml`. `umap-learn` is a core dependency. Optional extras: `torch`, `tf`, `llm`, `nlp`.
- `common/` is an installed package. Use `from common.datasets import ...` and `from common.plotting import ...`. Datasets come from sklearn built-ins, seeded generators, or download-and-cache into the gitignored `data/` folder.
- Notebooks are committed with outputs stripped (`nbstripout`). Each stays under 30KB.
- Prose depth is one markdown cell per code cell, about 500 words per notebook. Sections 01 and 02 got the deeper blog-quality pass in `polish261114`.
- Source folders under `/home/anthony/github_backup/` are read-only. Their `.git` directories were removed earlier at the user's request.
- Section 07 drops Unsloth and bitsandbytes 4-bit (CUDA and Linux only) and rewrites the engine on plain `transformers` + `peft` + `Trainer`, keeping the source's shape (config dataclass, mode-aware instruct/pretrain, data-prep helpers). LoRA comes from `peft` directly. The Ollama synthetic Q&A, PDF and vision parsing, GGUF export, chat CLI and Streamlit app were dropped; they need external services or heavy dependencies and are not needed to teach fine-tuning.
- Section 07 runs offline on a tiny GPT-2 from config for tests and nbmake; a real run is opt-in via `LLM_BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct`. `train()` forces CPU for the tiny model (deterministic, dodges the MPS float64 issue) and uses `get_device()` for a real model. Network-gated notebook cells catch only `(OSError, ImportError)` and print a skip note, so nbmake passes offline.

## Next steps

1. **Review and merge `polish261114` to main.** Ask the user to confirm before merging or pushing. The PR is open against `main`. This branch finished the deferred prose pass: 01 and 02 got the blog-quality depth pass, the 11 main-path section-05 notebooks were trimmed toward 500 words, and notebook 11 was rewritten onto the adding problem so the LSTM now beats the vanilla RNN on every one of five seeds (mean margin 83 points, CPU-deterministic).
2. Optional later work: a voice read-through of sections 03, 04, 06 and 07 notebook prose (this pass covered their READMEs, not every notebook), and any blog-post-specific editing as the posts are written.

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
uv sync --extra torch --extra llm
uv run --extra torch --extra llm pytest 07_llm/ common/
```

Source backup is at `/Volumes/SSD_SAM/github_backup/` (external SSD, must be mounted).

Optional, only to run the TensorFlow appendix:

```
UV_PROJECT_ENVIRONMENT=.venv-tf uv sync --extra torch --extra tf
UV_PROJECT_ENVIRONMENT=.venv-tf uv run --extra torch --extra tf pytest 05_deep_learning/appendix
```

Then ask Claude to read `CLAUDE.md` and this file. All seven sections and the final polish are in; the open work is reviewing and merging `polish261114`, then blog-post editing. Workflow in use: plan, confirm, sub-agents per notebook run sequentially or in parallel where independent, each owns its tests, coordinator re-runs the gates before committing.
