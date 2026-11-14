# Build-time notes: section 07 LLM

Minor assumptions recorded during the build. Major scope or design concerns go to
QUESTIONS.md under "Build-time questions" and stop the build.

## Step 1: env and skeleton

- Added `accelerate` to the `llm` extra, as the plan requires for `transformers.Trainer`.
- Package lives at `07_llm/llm_finetune/`; `07_llm/conftest.py` puts the section dir on
  `sys.path` so `import llm_finetune` resolves in tests.
- Clash check passed: with `--extra torch --extra llm` synced, `umap`, `transformers`
  5.22.2, `peft` 0.21.2, `datasets` and `accelerate` all import together. The known TF/umap
  LLVM conflict does not apply to the HF stack.

## Build details

- transformers 5.22.2 (major version): `TrainingArguments` uses `eval_strategy` and `use_cpu`,
  not the 4.x `evaluation_strategy`/`no_cuda`; no `use_mps_device`. Code written against the
  installed API.
- `train()` forces `use_cpu=True` when `config.base_model == "tiny"` so the teaching suite is
  deterministic and dodges the MPS float64 issue; a real model (Qwen) uses `get_device()`.
- Tiny tokenizer: a ByteLevel BPE trained with the `tokenizers` library on `demo_corpus()`,
  wrapped in `PreTrainedTokenizerFast`. transformers 5.x passes a `direction` kwarg to the
  backend's `enable_truncation`, so the model builder uses the underlying `tokenizers.Tokenizer`
  rather than the `ByteLevelBPETokenizer` wrapper. Tiny LoRA slice is 4096/152512 = 2.69%.
- nbmake runs each notebook with its own directory (`07_llm`) as cwd, so `import llm_finetune`
  resolves without sys.path edits in the notebooks.
- Ruff lints notebooks too, so notebook import blocks are kept sorted (`ruff check --fix`).
- `intuition_test.ipynb` uses a per-question predict/run/answer layout rather than section 06's
  pooled-answers-at-end. Both match the house "questions first, predictions, answers after"
  rule; the per-question form keeps each check next to its question.
