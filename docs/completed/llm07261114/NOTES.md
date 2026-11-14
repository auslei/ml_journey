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

## Review round 1

Cursor and Codex review of PR #2. All items addressed, no pushback.

Should-fix:
1. intuition Q2 now encodes per word (`sum` of `bpe.encode(w, merges)`), matching the
   word-level API from notebook 01. words 4, subwords 18, chars 25: BPE sits between.
2. `evaluate.generate` sets `model.eval()` and runs under `torch.no_grad()`, restoring the
   prior training mode in a `finally`. Greedy decode is now truly eval behaviour.
3. `test_evaluate` no longer decodes then re-encodes. Split into a string check and a
   token-budget check on `output_ids.shape[1] - input_ids.shape[1]`.
4. `07_llm/conftest.py` sets `collect_ignore_glob` for `*.ipynb` and `tests/*` when
   `transformers`/`peft` are absent, so bare `uv run pytest` stays green on a core install.
5. intuition Q5 answer drops the run-specific "511 to 333" figures; points at the printed
   before/after values and says perplexity falls by a clear margin.
6. notebook 02 datasets `except` drops `ValueError`, catching `(OSError, ImportError)` only.

Nits:
- intuition Q1 answer tied to the printed `pieces` (still `ne`, `wer`, end marker on this run).
- `model.load_base` hub `except` drops `ValueError`, so a bad config no longer silently falls
  back to the tiny model.
- train.py `eval_strategy`: chose the simpler path, notebook 04 now states the smoke run does
  not score the validation split and that the before/after perplexity is the evidence.
- root README testing block mentions `--extra torch --extra llm --extra nlp` for the full suite.
- intuition imports cell drops `warnings.filterwarnings("ignore")`.

Codex inline:
A. bpe `_merge_pair` rewritten to scan the symbol list with whole-symbol comparison (same scan
   as `encode`), so a merge never matches inside a fused symbol. Regression test on
   `['ccaacc', 'cccabcaa']`: with 100 merges each word now fuses to one token and round-trips;
   the buggy version left `ccaacc` at five tokens.
B. notebook 04 real-run path formats the prompt with `real_tok.apply_chat_template(...,
   add_generation_prompt=True)` when the tokenizer has a chat template, matching how
   `format_instruct` built the training data; the plain prompt stays for the fallback tokenizer.

## Review round 2

Cursor confirmed all round 1 and Codex findings resolved. Remaining items, all addressed,
no pushback.

Should fix:
- README.md roadmap/setup already marked section 07 done, but Layout showed `07_llm/ (planned)`
  and Future plans still treated 07 as unbuilt. Removed `(planned)` from the Layout tree,
  dropped the "Section 07, LLMs" Future plans item, and reworded the read-through item from
  "once section 07 lands" to "now that all seven sections are in".

Nits:
1. test_evaluate budget test now goes through `evaluate.generate_ids`, the token-level core
   `generate` wires `max_new_tokens` into, instead of calling `model.generate` directly. Pulled
   the id-producing body of `generate` out into a public `generate_ids(...) -> (input_ids,
   output_ids)` helper; `generate` is now a thin decode wrapper over it. The test asserts
   `output_ids.shape[1] - input_ids.shape[1] <= max_new_tokens`, so it exercises the real
   wiring and the eval/no_grad/restore path.
2. evaluate.py module docstring corrected: perplexity casts to float32 and leaves the model in
   eval mode; generate restores whatever training mode it found (no longer claims both leave it
   in eval mode).
3. intuition Q1 answer drops the redundant "plus the end marker" clause, since the printed
   `pieces` already include `</w>`.
