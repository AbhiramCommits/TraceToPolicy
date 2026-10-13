"""Hardware-aware model selection.

The central rule of this repo: never ship a config the machine cannot actually
train. ``resolve_model_config`` probes the runtime (CUDA VRAM, Apple MPS, or
plain CPU) and returns the *largest* tier that will train here:

===============  ===============================  ==========================
tier             base model                       adaptation
===============  ===============================  ==========================
``cuda_24gb``    Qwen/Qwen2.5-7B-Instruct          QLoRA (4-bit)
``cuda_small``   Qwen/Qwen2.5-1.5B-Instruct        LoRA
``cpu_apple``    Qwen/Qwen2.5-0.5B-Instruct        LoRA, reduced steps
===============  ===============================  ==========================

The resolved config is written into every results/training artifact so a reader
can tell exactly what produced a number.
"""

from __future__ import annotations

import os
import platform
from dataclasses import asdict, dataclass, field


@dataclass
class ModelConfig:
    """A concrete, trainable-on-this-machine configuration."""

    tier: str
    base_model: str
    load_in_4bit: bool
    device: str
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: list[str]
    max_steps_sft: int
    max_steps_dpo: int
    per_device_batch_size: int
    grad_accum: int
    learning_rate_sft: float
    learning_rate_dpo: float
    max_seq_length: int
    max_new_tokens: int
    notes: str = ""
    fallbacks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


_TIERS: dict[str, dict] = {
    "cuda_24gb": dict(
        base_model="Qwen/Qwen2.5-7B-Instruct",
        load_in_4bit=True,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        max_steps_sft=120,
        max_steps_dpo=60,
        per_device_batch_size=2,
        grad_accum=8,
        learning_rate_sft=2e-4,
        learning_rate_dpo=5e-5,
        max_seq_length=1024,
        max_new_tokens=256,
        notes="CUDA >=24GB: 7B with QLoRA (4-bit).",
    ),
    "cuda_small": dict(
        base_model="Qwen/Qwen2.5-1.5B-Instruct",
        load_in_4bit=False,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        max_steps_sft=120,
        max_steps_dpo=60,
        per_device_batch_size=4,
        grad_accum=4,
        learning_rate_sft=2e-4,
        learning_rate_dpo=5e-5,
        max_seq_length=1024,
        max_new_tokens=256,
        notes="CUDA <24GB: 1.5B with LoRA.",
    ),
    "cpu_apple": dict(
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        load_in_4bit=False,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        max_steps_sft=60,
        max_steps_dpo=30,
        per_device_batch_size=2,
        grad_accum=2,
        learning_rate_sft=3e-4,
        learning_rate_dpo=1e-4,
        max_seq_length=768,
        max_new_tokens=96,
        notes="CPU / Apple Silicon: 0.5B with LoRA, reduced steps.",
    ),
}

# Small models are fallbacks if a download fails. Order: largest -> smallest.
_FALLBACK_ORDER = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
]

_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Overridable from the environment so CI / tests can pin a tier.
_ENV_TIER = "TRACETOPOLICY_TIER"
_ENV_MODEL = "TRACETOPOLICY_BASE_MODEL"


def _detect_tier() -> str:
    """Return one of ``cuda_24gb``, ``cuda_small``, ``cpu_apple``."""
    forced = os.environ.get(_ENV_TIER)
    if forced in _TIERS:
        return forced

    try:
        import torch

        if torch.cuda.is_available():
            total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return "cuda_24gb" if total_gb >= 24 else "cuda_small"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "cpu_apple"
    except Exception:
        pass
    return "cpu_apple"


def resolve_model_config(force_tier: str | None = None) -> ModelConfig:
    """Probe the hardware and return the largest trainable configuration."""
    tier = force_tier or _detect_tier()
    if tier not in _TIERS:
        tier = "cpu_apple"
    spec = dict(_TIERS[tier])

    base_model = os.environ.get(_ENV_MODEL) or spec["base_model"]

    # Device selection: CUDA when present, else CPU. We deliberately avoid Apple
    # MPS for long batched generation/training: its caching allocator fragments
    # under varying sequence lengths and never returns the memory, which pushes
    # the host into swap on long harvests. CPU is a little slower but stable.
    device = os.environ.get("TRACETOPOLICY_DEVICE")
    if not device:
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    fallbacks = [m for m in _FALLBACK_ORDER if m != base_model]

    return ModelConfig(
        tier=tier,
        base_model=base_model,
        load_in_4bit=bool(spec["load_in_4bit"]),
        device=device,
        lora_r=spec["lora_r"],
        lora_alpha=spec["lora_alpha"],
        lora_dropout=spec["lora_dropout"],
        lora_target_modules=list(_TARGET_MODULES),
        max_steps_sft=spec["max_steps_sft"],
        max_steps_dpo=spec["max_steps_dpo"],
        per_device_batch_size=spec["per_device_batch_size"],
        grad_accum=spec["grad_accum"],
        learning_rate_sft=spec["learning_rate_sft"],
        learning_rate_dpo=spec["learning_rate_dpo"],
        max_seq_length=spec["max_seq_length"],
        max_new_tokens=spec["max_new_tokens"],
        notes=spec["notes"],
        fallbacks=fallbacks,
    )


def hardware_summary() -> dict:
    """Human-readable hardware probe, recorded in artifacts."""
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["mps_available"] = bool(
            getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        )
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda_vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
            )
    except Exception as exc:  # pragma: no cover - torch always importable in practice
        info["torch_error"] = str(exc)
    return info


if __name__ == "__main__":  # pragma: no cover
    import json

    print(
        json.dumps(
            {"config": resolve_model_config().to_dict(), "hardware": hardware_summary()}, indent=2
        )
    )
