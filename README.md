# ml_journey

A single learning path from linear algebra to LLM fine-tuning. It is my personal refresher and the training material behind a series of blog posts, so the explanations matter as much as the code.

Every notebook follows the same order: intuition, maths, cost function, a from-scratch version, the library version, tuning, and what breaks. Prose is short. The code runs end to end from a clean checkout, every random call is seeded, and outputs are stripped before commit.

## Roadmap

| Section | Topic | State |
|---|---|---|
| [01](01_linear_algebra_calculus) | Linear algebra and calculus, with autodiff built from scratch | done |
| [02](02_probability) | Probability, Bayes, and the MLE/MAP bridge to ML | done |
| [03](03_classical_ml) | Regression, classification, trees, ensembles, metrics | done |
| [04](04_unsupervised) | k-means, hierarchical, DBSCAN, GMM, PCA, t-SNE, UMAP, evaluation | done |
| [05](05_deep_learning) | Neural nets, optimisers, CNNs, transfer learning, RNNs, VAE | done |
| [06](06_nlp) | NLP: preprocessing, embeddings, topic models, sequence models, attention | done |
| [07](07_llm) | LLMs: tokenisation, HuggingFace basics, LoRA, a fine-tuning walkthrough | done |

Each section has its own README with a concept map, a reading order and a list of what you should be able to explain at the end. Each also closes with an `intuition_test.ipynb`: questions first, your predictions written down before running the code, answers after.

## Setup

You need [uv](https://docs.astral.sh/uv/) and Python 3.12.

```
git clone https://github.com/auslei/ml_journey.git
cd ml_journey
uv sync                             # sections 01 to 04
uv sync --extra torch               # adds section 05 (PyTorch, torchvision, Keras 3)
uv sync --extra torch --extra nlp   # adds section 06 (gensim, NLTK, spaCy)
uv sync --extra torch --extra llm   # adds section 07 (transformers, peft, datasets, accelerate)
```

Section 06 also needs the spaCy English model and a few NLTK corpora. Install the model once with `uv run --extra nlp python -m spacy download en_core_web_sm`. The NLTK data (punkt, stopwords, gutenberg, wordnet, movie_reviews) downloads automatically on first run.

Section 07 runs offline by default on a tiny GPT-2 built from a small config, so the tests and notebooks need no network. A real fine-tune is opt-in: set `LLM_BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct` to download Qwen (about 1GB) into the Hugging Face cache and run the same pipeline on MPS or CPU. The `llm` extra is independent of the `nlp` extra, and no TensorFlow is involved.

Open the notebooks in VS Code or any Jupyter front end, using the `.venv` kernel (`ipykernel` is installed).

Datasets come from scikit-learn, seeded generators, or a download that is cached in the gitignored `data/` folder. The first run of the deep learning notebooks downloads MNIST and CIFAR-10 (about 170MB for CIFAR-10).

### Deep learning notes

- Keras 3 is the teaching API and runs on the PyTorch backend. Native PyTorch is used where the training loop is the lesson.
- `common.dl.get_device()` picks CUDA, then Apple MPS, then CPU. Everything runs on CPU. The numbers quoted in section 05 were read off a CUDA machine and move by a few points elsewhere.
- The TensorFlow appendix (`tf.data` and `GradientTape`) needs its own environment, because TensorFlow and `umap` cannot share one:

```
UV_PROJECT_ENVIRONMENT=.venv-tf uv sync --extra torch --extra tf
UV_PROJECT_ENVIRONMENT=.venv-tf uv run --extra torch --extra tf pytest 05_deep_learning/appendix
```

## Tests

Notebooks are tested by running them end to end.

```
uv run pytest                                         # core sections, about 6 minutes with section 05
uv run --extra torch --extra llm --extra nlp pytest   # also run sections 06 and 07
uv run ruff check .
uv run ruff format --check .
```

Sections 06 and 07 are skipped without their extras, so the bare `uv run pytest` stays green on a core install.

Shared code lives in `common/` (datasets, plotting, deep learning helpers) and has its own unit tests. Sections talk to each other only through `common/`.

## Future plans

1. **Section 07, LLMs.** Tokenisation, a concept notebook on LoRA, and a fine-tuning pipeline with real Python modules and tests. No model weights or datasets are committed.
2. **A deeper writing pass.** The early notebooks (01 and 02) need a blog-quality rewrite, and the section 05 notebooks run longer than the prose target.
3. **A better recurrent example.** In notebook 11 the LSTM does not beat the vanilla RNN on the delayed-recall task, and the result depends on the seed. A better task or training budget would show the textbook result.
4. **A full read-through.** One pass over all prose for voice and consistency once section 07 lands.

## Layout

```
01_linear_algebra_calculus/   05_deep_learning/
02_probability/               06_nlp/
03_classical_ml/              07_llm/            (planned)
04_unsupervised/              common/
```

Working rules for contributors and for Claude Code sessions are in [CLAUDE.md](CLAUDE.md). Current status and the handoff for the next session are in [PROGRESS.md](PROGRESS.md).
