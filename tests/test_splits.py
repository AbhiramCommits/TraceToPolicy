import pytest

from tracetopolicy.splits import assert_no_leakage, get_splits, load_tasks, split_summary


def test_split_sizes_and_disjoint():
    splits = get_splits()
    total = len(load_tasks())
    assert len(splits["train"]) + len(splits["dev"]) + len(splits["test"]) == total
    assert total >= 120
    assert not (set(splits["train"]) & set(splits["dev"]))
    assert not (set(splits["train"]) & set(splits["test"]))
    assert not (set(splits["dev"]) & set(splits["test"]))


def test_split_deterministic():
    assert get_splits() == get_splits()


def test_split_summary():
    s = split_summary()
    assert set(s) == {"train", "dev", "test"}


def test_assert_no_leakage_passes():
    assert_no_leakage(["a", "b"], ["c", "d"])


def test_assert_no_leakage_raises():
    with pytest.raises(AssertionError):
        assert_no_leakage(["a", "b"], ["b", "c"])
