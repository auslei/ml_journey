# Section 07 LLM: build plan

Feature `llm07261114`. Final section of the learning path: tokenisation, HuggingFace
basics, LoRA, and a fine-tuning walkthrough driven by real Python modules.

## Core design decision

The source `llm_finetune` is built on **Unsloth + bitsandbytes 4-bit quantisation**. That
stack is CUDA and Linux only. It does not run on Apple MPS or CPU, which is the target here.
So the plan keeps the *shape* of the source (config dataclass, a tuner class, data-prep
helpers, mode-aware instruct/pretrain) but rewrites the engine on plain
`transformers` + `peft` + `Trainer`. No Unsloth, no bitsandbytes, no TRL. LoRA comes from
`peft` directly. This is the single biggest adaptation and it is confirmed in QUESTIONS.md.

Two model paths run side by side:

- **Tests and nbmake (default, offline):** a tiny GPT-2 built from a small config in code,
  with a tiny BPE tokeniser trained on a local corpus. Zero download, fully offline,
  deterministic. The whole 07 suite stays under a minute.
- **Real run (opt-in, documented):** `Qwen2.5-0.5B-Instruct` on MPS or CPU, selected by the
  `LLM_BASE_MODEL` env var. Downloads once (about 1GB) into the HuggingFace cache. Never run
  by the test suite.

Model downloads cache in the standard HF cache (`~/.cache/huggingface`), not the repo. No
weights or datasets are committed. Network-dependent notebook cells are wrapped in
try/except that prints a skip note, so nbmake stays green offline.

## Notebooks

Four content notebooks plus README and intuition test. Each follows the house order where it
fits: intuition, maths, from-scratch, library version, tuning, what breaks.

| File | Purpose (one line) |
|---|---|
| `01_tokenisation.ipynb` | Byte-pair encoding by hand on a tiny corpus, then a HuggingFace tokenizer trained offline, then (network-gated) the Qwen tokenizer; why subword beats word or character. |
| `02_huggingface_basics.ipynb` | The HF trio adapted from `h0`/`h1`/`h2`: `pipeline()` for ready tasks, `load_dataset` for data, `AutoTokenizer`/`AutoModel` for tokens and hidden states. Network cells degrade gracefully offline. |
| `03_lora_from_scratch.ipynb` | Low-rank adaptation maths: a `LoRALinear` module in pure PyTorch, the `W + (alpha/r)·BA` update, the parameter-count saving, then the same thing with `peft.LoraConfig` on the tiny model. |
| `04_finetuning_walkthrough.ipynb` | Drives the `llm_finetune` package end to end: config, demo corpus, tiny model, attach LoRA, train, then perplexity and a sample generation before and after. Shows how to swap in Qwen for a real run. |
| `README.md` | Concept map, reading order, what you should be able to explain, how the offline and real-model paths differ. |
| `intuition_test.ipynb` | Predict first, then run. Questions across all four notebooks. |

## Module layout

Package lives inside the section at `07_llm/llm_finetune/` (the directory `07_llm` is not a
valid module name, so the importable package sits one level down). A `07_llm/conftest.py`
inserts the section directory on `sys.path` so `import llm_finetune` works in tests; notebooks
run with their own directory as cwd, so the same import works there. Reuses
`common.dl.get_device` and `common.dl.set_seed` rather than duplicating device logic.

Six modules, each small and single-purpose.

### `config.py`
```python
@dataclass
class TrainConfig:
    base_model: str = "tiny"  # "tiny" builds offline; else an HF model id
    mode: str = "instruct"  # "instruct" | "pretrain"
    max_seq_length: int = 128
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: list[str] | None = None
    learning_rate: float = 2e-4
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    val_split: float = 0.1
    seed: int = 42
    output_dir: str = "data/llm_runs"


def from_yaml(path: str | Path) -> TrainConfig: ...
def load_config(path: str | Path | None = None, **overrides: Any) -> TrainConfig: ...
```
Validates `mode` and raises on an unknown value.

### `bpe.py` (pure Python, from-scratch byte-pair encoding)
```python
def train_bpe(corpus: list[str], num_merges: int) -> list[tuple[str, str]]: ...
def encode(word: str, merges: list[tuple[str, str]]) -> list[str]: ...
def _pair_counts(vocab: dict[str, int]) -> Counter: ...  # internal
```
No dependencies beyond the standard library. Raises on `num_merges < 0`.

### `data.py`
```python
def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]: ...
def read_jsonl(path: str | Path) -> list[dict]: ...
def write_jsonl(records: list[dict], path: str | Path) -> int: ...
def format_instruct(example: dict, tokenizer: Any) -> str: ...  # Alpaca or chat template
def format_pretrain(example: dict) -> str: ...
def build_dataset(records: list[dict], tokenizer: Any, config: TrainConfig) -> Any: ...
def demo_corpus() -> list[dict]: ...  # tiny built-in instruct set, offline
```
`chunk_text` is character-based with stdlib only (no langchain). `build_dataset` tokenises and
train/val splits deterministically from the seed. Raises when `overlap >= chunk_size`.

### `model.py`
```python
def tiny_causal_lm(seed: int = 42) -> tuple[Any, Any]: ...  # GPT-2 from small config, offline
def load_base(model_id: str, device: str | None = None) -> tuple[Any, Any]: ...
def add_lora(model: Any, config: TrainConfig) -> Any: ...  # peft get_peft_model
def resolve_target_modules(model: Any, config: TrainConfig) -> list[str]: ...
```
`tiny_causal_lm` builds a small `GPT2LMHeadModel` from a `GPT2Config` plus a tiny trained
tokenizer, no network. `load_base` falls back to `tiny_causal_lm` with a logged warning when
`model_id == "tiny"` or when the download fails offline. `resolve_target_modules` picks
`c_attn` for GPT-2, `q_proj`/`v_proj` for Qwen, so LoRA attaches on either architecture.

### `train.py`
```python
def train(model: Any, tokenizer: Any, dataset: Any, config: TrainConfig) -> dict[str, float]: ...
```
Wraps `transformers.Trainer` + `TrainingArguments`. Returns `{"train_loss", "steps",
"trainable_params", "total_params"}`. Logs an info line before each major step. Keeps dtype
float32 to dodge the MPS float64 problem seen in section 05.

### `evaluate.py`
```python
def perplexity(model: Any, tokenizer: Any, texts: list[str], device: str) -> float: ...
def generate(
    model: Any, tokenizer: Any, prompt: str, max_new_tokens: int = 50, seed: int = 42
) -> str: ...
```
`perplexity` is `exp(mean token loss)`. `generate` is seeded and deterministic for a fixed
seed.

### Dropped from the source
Ollama synthetic Q&A generation, PDF and vision parsing, GGUF export, the Streamlit app and the
chat CLI. They pull heavy or external dependencies (a running Ollama server, `pdfplumber`,
`pdf2image`) and are not needed to teach the fine-tuning path. Confirmed in QUESTIONS.md.

## Tests (`07_llm/tests/`)

Deterministic, tiny-model only, offline. Both branches of each two-outcome function covered.

| File | Covers |
|---|---|
| `test_config.py` | Defaults; `from_yaml` round-trip; `load_config` overrides win; unknown `mode` raises. |
| `test_bpe.py` | A known merge happens on a crafted corpus; single-character fallback when no merges; `encode` round-trips; `num_merges < 0` raises. |
| `test_data.py` | `chunk_text` normal split and the `overlap >= chunk_size` raise; `write_jsonl`/`read_jsonl` round-trip; `format_instruct` shape; `demo_corpus` non-empty and well-formed. |
| `test_model.py` | `tiny_causal_lm` forwards a batch; `add_lora` makes trainable params a small fraction of total (assert `trainable < total`); `resolve_target_modules` picks the GPT-2 module. |
| `test_train.py` | One-epoch train on tiny model and a 4-row dataset returns finite loss and the expected keys; trainable-param count matches the LoRA config. |
| `test_evaluate.py` | `perplexity` is finite and positive; `generate` returns a non-empty string no longer than the token budget. |

Gate: `uv run --extra torch --extra llm pytest 07_llm/` green, plus `ruff check` and
`ruff format --check` clean, plus the four notebooks pass `--nbmake` offline.

## Dependency changes

Current `llm` extra: `transformers`, `peft`, `datasets`.

- **Add `accelerate`.** `transformers.Trainer` needs it. `tokenizers` and `huggingface-hub`
  already ship with `transformers`, so no separate pins.
- No `unsloth`, `trl`, `bitsandbytes`, `sentencepiece` (Qwen2.5 uses a BPE tokenizer loaded
  through `tokenizers`), `ollama`, `pdfplumber` or `langchain`.
- Pin everything through `uv.lock` so the notebooks still run next year.

Clash check (a build-order step, not an assumption): run
`uv sync --extra torch --extra llm` in the main `.venv`, then confirm `import umap` still
works and `import transformers, peft, datasets, accelerate` all import. The known conflict is
TensorFlow against umap (duplicate LLVM), not transformers against umap, so this is expected
to pass, but it gets verified before any notebook work starts.

## Model and runtime choices

| Path | Model | Where | Expected time on an M-series Mac |
|---|---|---|---|
| Tests | tiny GPT-2 from config | `07_llm/tests/` | whole suite under 30s |
| nbmake (offline) | tiny GPT-2; downloads skipped | the four notebooks | under 3 min total |
| nbmake (first run, online) | small pretrained models for `02` pipelines | notebook `02` | plus 1 to 3 min of one-time downloads, then cached |
| Real fine-tune (opt-in) | `Qwen2.5-0.5B-Instruct` | manual, `LLM_BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct` | about 1GB download once; roughly 5 to 15 min for a 200-row, 1-epoch LoRA on MPS, slower on CPU |

Offline behaviour: with no network, the tiny path runs unchanged, and the network-gated cells
in `01` and `02` print a one-line skip note instead of erroring. nbmake passes either way.

## Risks

1. **Unsloth removal changes the engine.** The source training loop does not transfer. The
   rewrite on plain `Trainer` is new code and carries the most risk. Mitigated by the tiny-model
   train test running a real step.
2. **`peft` target modules differ by architecture.** GPT-2 uses `c_attn`; Qwen uses
   `q_proj`/`v_proj`. `resolve_target_modules` handles both; tested for GPT-2.
3. **MPS float64.** Section 05 hit a float64/MPS failure. Keep tensors float32 throughout.
4. **nbmake offline flakiness.** Every download cell needs a try/except skip, or nbmake fails
   with no network. This is a notebook-authoring discipline, flagged for each sub-agent.
5. **`transformers`/`datasets` API churn.** `TrainingArguments` fields move between versions.
   Pin via `uv.lock` and write against the installed version, not from memory.
6. **umap clash (low).** Expected to pass; verified in build step 1 before anything depends on it.
7. **Import path for a digit-prefixed section.** Handled by `07_llm/conftest.py` and the
   notebook cwd; verified early.

## Build order (for sub-agents)

Sequential where a later step needs an earlier one; parallel where marked.

1. **Env and skeleton (sequential, first).** Add `accelerate` to the `llm` extra,
   `uv sync --extra torch --extra llm`, run the clash check (`import umap` plus the HF imports),
   create `07_llm/` with `llm_finetune/__init__.py`, `conftest.py`, `tests/`, and confirm
   `import llm_finetune` resolves. Owns: dependency change, import plumbing.
2. **`bpe.py` + `config.py` + their tests (parallel, independent of each other).** No torch
   needed for bpe; config is a dataclass. Each sub-agent writes and passes its tests.
3. **`data.py` + tests (after 2).** Depends on `config.TrainConfig`.
4. **`model.py` + `train.py` + `evaluate.py` + tests (after 3).** Depends on config and data.
   One sub-agent owns the training trio to keep the `Trainer` contract consistent.
5. **Notebook `01_tokenisation` (after `bpe.py`) and `02_huggingface_basics` (after step 1),
   in parallel.** 01 drives `bpe.py`; 02 is standalone HF material with offline gates.
6. **Notebook `03_lora_from_scratch` (after `model.py`) and `04_finetuning_walkthrough`
   (after step 4), in parallel.** 04 drives the whole package.
7. **README + intuition_test (after all notebooks).**
8. **Integration (sequential, last, coordinator).** Run `07_llm/` suite, full repo suite,
   `ruff check`, `ruff format --check`, all four notebooks under `--nbmake`. Then update root
   `README.md` roadmap and layout block, `PROGRESS.md` status and next steps. Commit on the
   feature branch, ask before merge.

Per the house workflow, each sub-agent owns its tests and runs the section suite plus ruff
before reporting done; the coordinator re-runs the gates after every sub-agent and again at
integration.

## Review refinements (Engineer Manager, 2026-11-14)

- Accept the defaults in QUESTIONS.md 1 to 5 unless Anthony overrides them.
- Section 06 review (PR #1) moves gensim, nltk and spacy into a new `nlp` extra. Rebase this branch onto `main` after PR #1 merges, before build step 1, and keep the `llm` extra independent of `nlp`.
- Notebook 04 must state plainly that the tiny offline model produces nonsense text, and that before/after perplexity is the lesson. Quote Qwen numbers in prose only from a real run; otherwise describe what to expect without figures.
- Network-gated cells: catch only network or HF hub errors (for example `OSError`, `requests` errors), not a bare `except`, so real bugs still fail nbmake.
- Em dashes: zero in 07 prose, per `~/dev/my_voice.md`.
