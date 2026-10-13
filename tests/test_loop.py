import json

from tracetopolicy.loop import (
    _degrade_datasets,
    _get_version,
    _write_jsonl,
    load_registry,
    save_registry,
)


def test_registry_roundtrip(tmp_path):
    p = str(tmp_path / "reg.json")
    reg = {
        "current": "v1",
        "versions": [{"version": "v0", "adapter": None}, {"version": "v1", "adapter": "x"}],
    }
    save_registry(reg, p)
    loaded = load_registry(p)
    assert loaded["current"] == "v1"
    assert len(loaded["versions"]) == 2


def test_load_registry_missing_returns_empty(tmp_path):
    reg = load_registry(str(tmp_path / "nope.json"))
    assert reg["current"] == "v0" and reg["versions"] == []


def test_get_version():
    reg = {"versions": [{"version": "v0", "adapter": None}]}
    assert _get_version(reg, "v0")["version"] == "v0"
    assert _get_version(reg, "v9")["version"] == "v0"


def test_degrade_datasets_inverts_preferences(tmp_path):
    sft = tmp_path / "sft.jsonl"
    dpo = tmp_path / "dpo.jsonl"
    _write_jsonl(
        str(sft),
        [
            {
                "task_id": "t1",
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "SELECT ok"},
                ],
            }
        ],
    )
    _write_jsonl(
        str(dpo),
        [
            {
                "task_id": "t1",
                "prompt": [],
                "chosen": [{"role": "assistant", "content": "SELECT ok"}],
                "rejected": [{"role": "assistant", "content": "SELECT bad"}],
            }
        ],
    )

    bad_sft, bad_dpo = _degrade_datasets(str(sft), str(dpo), str(tmp_path))
    assert json.loads(open(bad_sft).readline())["messages"][-1]["content"] == "SELECT 1"
    row = json.loads(open(bad_dpo).readline())
    assert row["chosen"][0]["content"] == "SELECT bad"
    assert row["rejected"][0]["content"] == "SELECT ok"
