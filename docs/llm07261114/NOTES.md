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
