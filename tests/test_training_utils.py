import json
import os

from tracetopolicy.model_config import resolve_model_config
from tracetopolicy.training_utils import (
    file_hash,
    load_yaml,
    make_dataset,
    pick,
    save_model_card,
    setup_wandb,
    timer,
    write_json,
)


def test_pick_uses_fallback_on_none():
    assert pick(None, 7) == 7
    assert pick(3, 7) == 3


def test_write_and_load_yaml_json(tmp_path):
    p = tmp_path / "obj.json"
    write_json(str(p), {"a": 1, "b": [1, 2]})
    assert json.loads(p.read_text())["a"] == 1
    y = tmp_path / "c.yaml"
    y.write_text("seed: 3\nname: x\n")
    assert load_yaml(str(y))["seed"] == 3


def test_file_hash_stable(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("hello")
    assert file_hash(str(p)) == file_hash(str(p))


def test_timer_returns_minutes():
    with timer() as elapsed:
        pass
    assert elapsed() >= 0


def test_setup_wandb_offline(monkeypatch):
    monkeypatch.delenv("WANDB_MODE", raising=False)
    report = setup_wandb("proj", "run")
    assert report in ("wandb", "none")
    assert os.environ["WANDB_MODE"] == "offline"


def test_make_dataset_keeps_columns(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text(json.dumps({"messages": [{"role": "user", "content": "hi"}], "extra": 1}) + "\n")
    ds = make_dataset(str(p), keep_columns=["messages"])
    assert ds.column_names == ["messages"]


def test_save_model_card(tmp_path):
    cfg = resolve_model_config(force_tier="cpu_apple")
    ds = tmp_path / "d.jsonl"
    ds.write_text("{}\n")
    card = save_model_card(
        str(tmp_path),
        stage="sft",
        model_cfg=cfg,
        lora_cfg={"r": 8, "lora_alpha": 16, "lora_dropout": 0.05, "target_modules": ["q_proj"]},
        steps=10,
        seed=1,
        dataset_path=str(ds),
        wall_minutes=1.23,
    )
    assert card["stage"] == "sft" and card["lora_rank"] == 8
    assert (tmp_path / "model_card.json").exists()
