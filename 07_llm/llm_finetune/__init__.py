"""Small, offline-first LoRA fine-tuning package for section 07.

Keeps the shape of the original ``llm_finetune`` (config dataclass, mode-aware
instruct/pretrain, data-prep helpers) but runs on plain ``transformers`` + ``peft`` +
``Trainer`` so it works on Apple MPS and CPU. No Unsloth, no bitsandbytes.
"""
