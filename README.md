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
| 06 | NLP: embeddings, topic models, language models, attention | planned |
| 07 | LLMs: tokenisation, fine-tuning, LoRA | planned |

Each section has its own README with a concept map, a reading order and a list of what you should be able to explain at the end. Each also closes with an `intuition_test.ipynb`: questions first, your predictions written down before running the code, answers after.

## Setup

You need [uv](https://docs.astral.sh/uv/) and Python 3.12.

```
git clone https://github.com/auslei/ml_journey.git
cd ml_journey
uv sync                      # sections 01 to 04
uv sync --extra torch        # adds section 05 (PyTorch, torchvision, Keras 3)
```

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
uv run pytest                # about 6 minutes with section 05
uv run ruff check .
uv run ruff format --check .
```

Shared code lives in `common/` (datasets, plotting, deep learning helpers) and has its own unit tests. Sections talk to each other only through `common/`.

## Future plans

1. **Section 06, NLP.** Word embeddings (CBOW and skip-gram), Word2Vec, topic models (LDA), autoregressive language models, RNNs on text, and a short attention and transformer notebook. All in native PyTorch.
2. **Section 07, LLMs.** Tokenisation, a concept notebook on LoRA, and a fine-tuning pipeline with real Python modules and tests. No model weights or datasets are committed.
3. **A deeper writing pass.** The early notebooks (01 and 02) need a blog-quality rewrite, and the section 05 notebooks run longer than the prose target.
4. **A better recurrent example.** In notebook 11 the LSTM does not beat the vanilla RNN on the delayed-recall task, and the result depends on the seed. A better task or training budget would show the textbook result.
5. **A full read-through.** One pass over all prose for voice and consistency once sections 06 and 07 land.

## Layout

```
01_linear_algebra_calculus/   05_deep_learning/
02_probability/               06_nlp/            (planned)
03_classical_ml/              07_llm/            (planned)
04_unsupervised/              common/
```

Working rules for contributors and for Claude Code sessions are in [CLAUDE.md](CLAUDE.md). Current status and the handoff for the next session are in [PROGRESS.md](PROGRESS.md).
