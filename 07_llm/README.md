# 07 LLM

This section covers large language models from the tokeniser up to a LoRA fine-tune. The
thread is adaptation: how text becomes tokens, how a pre-trained model is loaded, and how a
small low-rank update tunes it without touching the base weights. Everything runs on plain
`transformers` + `peft` + `Trainer`, so the code works on Apple MPS and CPU. No Unsloth, no
bitsandbytes.

## Two model paths

The section runs offline by default on a tiny GPT-2 built from a small config in code, with a
tiny BPE tokeniser trained on a local corpus. Zero download, deterministic, the whole suite
finishes in seconds. The tiny model is for mechanics, not quality: it produces nonsense text
no matter how low the loss goes.

A real run is opt-in. Set `LLM_BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct` and the same pipeline
loads Qwen on MPS or CPU. That downloads about 1GB once into the Hugging Face cache and takes
a few minutes to train a small LoRA. The test suite and nbmake never touch it. Network-gated
cells catch only hub or load errors and print a skip note, so the notebooks pass with no
network.

## Concept map

```
raw text
    |
    v
tokenisation: byte-pair encoding, subword between word and character (01)
    |
    v
pre-trained model loaded through HuggingFace
    |--- pipeline(): a task in one line (02)
    |--- load_dataset: standardised data (02)
    |--- AutoTokenizer / AutoModel: tokens and hidden states (02)
    |
    v
adaptation
    |--- LoRA: freeze W, learn a low-rank update B A, train a small fraction (03)
    |--- the W + (alpha/r) B A update, peft.LoraConfig on the model (03)
    |
    v
fine-tune end to end: config, data, attach LoRA, train, measure perplexity (04)
```

## Reading order

1. `01_tokenisation.ipynb`: byte-pair encoding by hand on a tiny corpus, then a HuggingFace
   tokenizer trained offline, then the Qwen tokenizer (network-gated). Why subword beats word
   or character, and why an unseen word still encodes with no out-of-vocabulary token.
2. `02_huggingface_basics.ipynb`: the three levels of HuggingFace. `pipeline()` for ready
   tasks, `load_dataset` for data, `AutoTokenizer`/`AutoModel` for tokens and hidden states.
   Every network cell degrades gracefully offline.
3. `03_lora_from_scratch.ipynb`: a `LoRALinear` module in pure PyTorch, the
   `W + (alpha/r) B A` update, the parameter-count saving, then the same thing with
   `peft.LoraConfig` on the tiny model.
4. `04_finetuning_walkthrough.ipynb`: drives the `llm_finetune` package end to end. Config,
   demo corpus, tiny model, attach LoRA, train, then perplexity and a sample generation
   before and after. Shows how to swap in Qwen for a real run.
5. `intuition_test.ipynb`: predict first, then run. Questions covering all four notebooks.

Dependencies: `uv sync --extra torch --extra llm` (the `llm` extra pulls transformers, peft,
datasets and accelerate). The `llm` extra is independent of the `nlp` extra. No TensorFlow
needed. The tiny path needs no network. The Qwen path downloads into the standard Hugging
Face cache on first use.

## The package

The reusable code lives at `07_llm/llm_finetune/` (the directory `07_llm` is not a valid module
name, so the package sits one level down). `07_llm/conftest.py` puts the section directory on
`sys.path` so `import llm_finetune` works in tests; notebooks run with their own directory as
cwd, so the same import works there.

- `config.py`: `TrainConfig` dataclass, loadable from YAML, validates the mode.
- `bpe.py`: from-scratch byte-pair encoding, standard library only.
- `data.py`: character chunking, JSONL IO, instruct and pretrain formatting, the demo corpus.
- `model.py`: the tiny GPT-2 builder, base-model loader with an offline fallback, LoRA attach,
  and target-module resolution (`c_attn` for GPT-2, `q_proj`/`v_proj` for Qwen).
- `train.py`: wraps `transformers.Trainer`, returns loss, steps and parameter counts.
- `evaluate.py`: perplexity as `exp(mean token loss)`, and seeded greedy generation.

The source `llm_finetune` project ran on Unsloth and bitsandbytes 4-bit, which are CUDA and
Linux only. The Ollama synthetic Q&A generation, the PDF and vision parsing, the GGUF export,
the chat CLI and the Streamlit app were dropped. They need external services or heavy
dependencies and are not needed to teach the fine-tuning path.

## What you should be able to explain at the end

- Why word-level tokenisation has an unbounded vocabulary and character-level makes sequences
  long, and how byte-pair encoding sits between the two.
- Why a byte-pair tokeniser never hits an out-of-vocabulary token: an unseen word falls back to
  smaller pieces, down to single characters.
- What a merge is in BPE, and why training stops once the merge budget runs out.
- The three levels of the HuggingFace stack, and why the tokenizer and model must share the
  same model id.
- What a hidden state is: the shape `(batch, sequence, hidden)` a model returns, and that each
  position carries a vector.
- The LoRA update `W + (alpha/r) B A`: why `B` starts at zero so the model is unchanged at
  init, and why only `A` and `B` are trained.
- Why LoRA trains a small fraction of the parameters: on the tiny GPT-2 the LoRA slice is about
  2.7% of the total, and the rest stays frozen.
- What rank `r` and `alpha` each control, and why target modules differ by architecture.
- What perplexity measures, and why a falling perplexity on the training data shows the LoRA
  update learned something even when the generated text is still gibberish from a tiny model.
- Why real gains need a real base model and more data, and why perplexity is only comparable
  under the same tokenizer.

This is the last section of the path. It closes the loop from linear algebra to a working,
if tiny, fine-tune.
