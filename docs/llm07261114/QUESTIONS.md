# Open questions for Anthony: section 07 LLM

These change what gets built. The plan assumes a default answer for each, stated below, so work
can start if you are happy with the defaults. Flag any you want changed.

## 1. Drop Unsloth for plain transformers + peft? (biggest one)

The source `llm_finetune` runs on Unsloth and bitsandbytes 4-bit. That stack is CUDA and Linux
only and will not run on Apple MPS or CPU, which is the target for this section.

**Default:** rewrite the engine on plain `transformers` + `peft` + `Trainer`, keep the source's
shape (config dataclass, mode-aware instruct/pretrain, data-prep helpers). LoRA from `peft`
directly.

Alternative: keep an Unsloth path as a documented CUDA-only appendix. More code, and you cannot
run it on this Mac, so I would not.

## 2. Default base model: Qwen2.5-0.5B-Instruct?

**Default:** `Qwen2.5-0.5B-Instruct` for the documented real run, tiny GPT-2 from config for
tests and nbmake.

Alternatives at a similar size: `SmolLM2-360M-Instruct`, `TinyLlama-1.1B`. Happy to switch if
you have a preference.

## 3. Drop the Ollama / PDF / vision data-prep from the source?

The source generates synthetic Q&A through a running Ollama server and parses PDFs with
`pdfplumber` and a vision model. That needs external services and heavy dependencies.

**Default:** keep the deterministic, testable helpers (chunking, JSONL IO, instruct/pretrain
formatting) and a small built-in demo corpus. Drop Ollama, PDF and vision. Mention the Ollama
synthetic-data idea in prose without code.

Alternative: port the Ollama path as an optional, untested notebook cell.

## 4. Real fine-tune strictly opt-in, never in the test suite?

**Default:** the test suite and nbmake only ever touch the tiny offline model. The Qwen run is
manual, behind the `LLM_BASE_MODEL` env var, and documented in the walkthrough notebook and the
README. This keeps the suite fast and offline-tolerant.

Confirm you do not want a real fine-tune running in CI or nbmake.

## 5. Also drop GGUF export, the chat CLI and the Streamlit app?

These exist in the source but are not needed to teach the fine-tuning path, and GGUF export in
particular pulls more tooling.

**Default:** out of scope for section 07. Say so if you want any of them included.
