"""DPO on the preference pairs, initialized from the SFT adapter.

The SFT model is the *frozen reference*: we build a second, frozen copy of the
base model carrying the SFT adapter and hand it to ``DPOTrainer`` as
``ref_model``. The policy continues training the SFT adapter weights.

Run: ``python -m tracetopolicy.train_dpo --config configs/dpo.yaml``
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


def _build_model(base_model: str, mcfg, load_in_4bit: bool):
    import torch
    from transformers import AutoModelForCausalLM

    dtype = torch.float16 if mcfg.device == "cuda" else torch.float32
    kwargs: dict = {"trust_remote_code": True, "dtype": dtype}
    if load_in_4bit and mcfg.device == "cuda":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        kwargs["device_map"] = "auto"
    return AutoModelForCausalLM.from_pretrained(base_model, **kwargs)


def main(config_path: str = "configs/dpo.yaml") -> dict:
    cfg = load_yaml(config_path)
    mcfg = resolve_model_config()
    if cfg.get("base_model"):
        mcfg.base_model = cfg["base_model"]

    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)

    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    from .trl_patches import patch_trl_for_cpu

    if patch_trl_for_cpu():
        print("[dpo] installed pure-PyTorch fused LM-head (no Triton on this platform)")

    torch.manual_seed(seed)

    sft_adapter = cfg.get("sft_adapter", "artifacts/sft")
    out_dir = cfg.get("output_dir", "artifacts/dpo")
    max_length = int(pick(cfg.get("max_length"), mcfg.max_seq_length))
    max_steps = int(pick(cfg.get("max_steps"), mcfg.max_steps_dpo))
    lr = float(pick(cfg.get("learning_rate"), mcfg.learning_rate_dpo))
    batch = int(pick(cfg.get("per_device_train_batch_size"), mcfg.per_device_batch_size))
    grad_accum = int(pick(cfg.get("gradient_accumulation_steps"), mcfg.grad_accum))
    load_in_4bit = bool(pick(cfg.get("load_in_4bit"), mcfg.load_in_4bit))
    beta = float(cfg.get("beta", 0.1))

    report = setup_wandb(cfg.get("wandb_project", "tracetopolicy"), cfg.get("run_name", "dpo"))
    dataset_path = cfg["dataset"]
    ds = make_dataset(dataset_path, keep_columns=["prompt", "chosen", "rejected"])
    print(
        f"[dpo] dataset={dataset_path} rows={len(ds)} tier={mcfg.tier} "
        f"device={mcfg.device} sft_adapter={sft_adapter}"
    )

    tokenizer = AutoTokenizer.from_pretrained(mcfg.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # policy: base + trainable SFT adapter
    policy_base = _build_model(mcfg.base_model, mcfg, load_in_4bit)
    policy = PeftModel.from_pretrained(policy_base, sft_adapter, is_trainable=True)
    policy.config.use_cache = False

    # reference: a second frozen copy of the base model with the SFT adapter
    # MERGED in, so it is a plain model (TRL patches a plain reference cleanly;
    # a PeftModel reference is patched incorrectly).
    ref_model = None
    try:
        ref_base = _build_model(mcfg.base_model, mcfg, load_in_4bit)
        ref_peft = PeftModel.from_pretrained(ref_base, sft_adapter)
        ref_model = ref_peft.merge_and_unload(safe_merge=True)
        for p in ref_model.parameters():
            p.requires_grad = False
        ref_model.eval()
        print("[dpo] using frozen (merged) SFT model as reference")
    except Exception as exc:  # noqa: BLE001
        print(f"[dpo] could not build explicit SFT reference ({exc}); using adapter-disabled base")
        ref_model = None

    args = DPOConfig(
        output_dir=out_dir,
        max_steps=max_steps,
        num_train_epochs=float(cfg.get("num_train_epochs", 4)),
        learning_rate=lr,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=grad_accum,
        beta=beta,
        loss_type=cfg.get("loss_type", "sigmoid"),
        max_length=max_length,
        logging_steps=int(cfg.get("logging_steps", 5)),
        save_strategy=cfg.get("save_strategy", "no"),
        report_to=[report],
        run_name=cfg.get("run_name", "dpo"),
        seed=seed,
        data_seed=seed,
        bf16=False,
        fp16=(mcfg.device == "cuda"),
        gradient_checkpointing=False,
        use_cpu=(mcfg.device == "cpu"),
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        precompute_ref_log_probs=bool(cfg.get("precompute_ref_log_probs", False)),
        warmup_steps=min(5, max(1, max_steps // 10)),
    )

    trainer = DPOTrainer(
        model=policy,
        ref_model=ref_model,
        args=args,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    with timer() as elapsed:
        result = trainer.train()
    wall = elapsed()

    os.makedirs(out_dir, exist_ok=True)
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    lora_cfg = {
        "r": mcfg.lora_r,
        "lora_alpha": mcfg.lora_alpha,
        "lora_dropout": mcfg.lora_dropout,
        "target_modules": list(mcfg.lora_target_modules),
    }
    # capture reward margins from the log history
    chosen_scores = [h.get("rewards/chosen") for h in trainer.state.log_history if "rewards/chosen" in h]
    rejected_scores = [h.get("rewards/rejected") for h in trainer.state.log_history if "rewards/rejected" in h]
    margins = [h.get("rewards/margins") for h in trainer.state.log_history if "rewards/margins" in h]

    card = save_model_card(
        out_dir,
        stage="dpo",
        model_cfg=mcfg,
        lora_cfg=lora_cfg,
        steps=max_steps,
        seed=seed,
        dataset_path=dataset_path,
        wall_minutes=wall,
        extra={
            "parent_adapter": sft_adapter,
            "beta": beta,
            "loss_type": cfg.get("loss_type", "sigmoid"),
            "num_examples": len(ds),
            "learning_rate": lr,
            "final_reward_margin": margins[-1] if margins else None,
            "mean_chosen_reward": (sum(chosen_scores) / len(chosen_scores)) if chosen_scores else None,
            "mean_rejected_reward": (sum(rejected_scores) / len(rejected_scores)) if rejected_scores else None,
            "training_loss": getattr(result, "training_loss", None),
        },
    )
    log = {
        "stage": "dpo",
        "config": cfg,
        "model_card": card,
        "log_history": trainer.state.log_history,
        "reward_margins": margins,
    }
    write_json(cfg.get("log_path", "results/train_dpo_log.json"), log)
    print(json.dumps(card, indent=2))
    return card


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dpo.yaml")
    args = parser.parse_args()
    main(args.config)
