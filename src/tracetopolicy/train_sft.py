"""LoRA / QLoRA supervised fine-tuning on the repair-aware SFT dataset.

Run: ``python -m tracetopolicy.train_sft --config configs/sft.yaml``
"""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np

from .model_config import resolve_model_config
from .training_utils import (
    load_yaml,
    make_dataset,
    pick,
    save_model_card,
    setup_wandb,
    timer,
    write_json,
)


def main(config_path: str = "configs/sft.yaml") -> dict:
    cfg = load_yaml(config_path)
    mcfg = resolve_model_config()
    if cfg.get("base_model"):
        mcfg.base_model = cfg["base_model"]

    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)

    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    torch.manual_seed(seed)

    lora_r = int(pick(cfg.get("lora_r"), mcfg.lora_r))
    lora_alpha = int(pick(cfg.get("lora_alpha"), mcfg.lora_alpha))
    lora_dropout = float(pick(cfg.get("lora_dropout"), mcfg.lora_dropout))
    max_seq_length = int(pick(cfg.get("max_seq_length"), mcfg.max_seq_length))
    max_steps = int(pick(cfg.get("max_steps"), mcfg.max_steps_sft))
    lr = float(pick(cfg.get("learning_rate"), mcfg.learning_rate_sft))
    batch = int(pick(cfg.get("per_device_train_batch_size"), mcfg.per_device_batch_size))
    grad_accum = int(pick(cfg.get("gradient_accumulation_steps"), mcfg.grad_accum))
    load_in_4bit = bool(pick(cfg.get("load_in_4bit"), mcfg.load_in_4bit))
    out_dir = cfg.get("output_dir", "artifacts/sft")

    report = setup_wandb(cfg.get("wandb_project", "tracetopolicy"), cfg.get("run_name", "sft"))
    dataset_path = cfg["dataset"]
    ds = make_dataset(dataset_path, keep_columns=["messages"])
    print(f"[sft] dataset={dataset_path} rows={len(ds)} tier={mcfg.tier} device={mcfg.device}")

    tokenizer = AutoTokenizer.from_pretrained(mcfg.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if mcfg.device == "cuda" else torch.float32
    model_kwargs: dict = {"trust_remote_code": True, "dtype": dtype}
    if load_in_4bit and mcfg.device == "cuda":
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(mcfg.base_model, **model_kwargs)
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(mcfg.lora_target_modules),
    )

    args = SFTConfig(
        output_dir=out_dir,
        max_steps=max_steps,
        num_train_epochs=float(cfg.get("num_train_epochs", 4)),
        learning_rate=lr,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=grad_accum,
        logging_steps=int(cfg.get("logging_steps", 5)),
        save_strategy=cfg.get("save_strategy", "no"),
        report_to=[report],
        run_name=cfg.get("run_name", "sft"),
        seed=seed,
        data_seed=seed,
        max_length=max_seq_length,
        bf16=False,
        fp16=(mcfg.device == "cuda"),
        gradient_checkpointing=False,
        use_cpu=(mcfg.device == "cpu"),
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        warmup_steps=min(5, max(1, max_steps // 10)),
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    with timer() as elapsed:
        result = trainer.train()
    wall = elapsed()

    os.makedirs(out_dir, exist_ok=True)
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    lora_cfg = {"r": lora_r, "lora_alpha": lora_alpha, "lora_dropout": lora_dropout, "target_modules": list(mcfg.lora_target_modules)}
    card = save_model_card(
        out_dir,
        stage="sft",
        model_cfg=mcfg,
        lora_cfg=lora_cfg,
        steps=max_steps,
        seed=seed,
        dataset_path=dataset_path,
        wall_minutes=wall,
        extra={
            "train_loss": getattr(result, "training_loss", None),
            "num_examples": len(ds),
            "learning_rate": lr,
            "batch_size": batch,
            "grad_accum": grad_accum,
            "max_length": max_seq_length,
        },
    )
    log = {
        "stage": "sft",
        "config": cfg,
        "model_card": card,
        "log_history": trainer.state.log_history,
    }
    write_json(cfg.get("log_path", "results/train_sft_log.json"), log)
    print(json.dumps(card, indent=2))
    return card


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sft.yaml")
    args = parser.parse_args()
    main(args.config)
