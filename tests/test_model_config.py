from tracetopolicy.model_config import (
    ModelConfig,
    hardware_summary,
    resolve_model_config,
)


def test_resolve_forced_tiers():
    for tier in ["cuda_24gb", "cuda_small", "cpu_apple"]:
        cfg = resolve_model_config(force_tier=tier)
        assert isinstance(cfg, ModelConfig)
        assert cfg.tier == tier
        assert cfg.base_model.startswith("Qwen/Qwen2.5")
        assert cfg.lora_r > 0
        assert cfg.max_steps_sft > 0


def test_cpu_tier_uses_small_model():
    cfg = resolve_model_config(force_tier="cpu_apple")
    assert cfg.base_model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert cfg.load_in_4bit is False
    assert cfg.fallbacks


def test_env_model_override(monkeypatch):
    monkeypatch.setenv("TRACETOPOLICY_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    cfg = resolve_model_config(force_tier="cpu_apple")
    assert cfg.base_model == "Qwen/Qwen2.5-1.5B-Instruct"


def test_to_dict_and_hardware_summary():
    cfg = resolve_model_config(force_tier="cpu_apple")
    d = cfg.to_dict()
    assert d["tier"] == "cpu_apple"
    hw = hardware_summary()
    assert "machine" in hw and "python" in hw
