import pytest

from tracetopolicy.cli import build_parser, main


@pytest.mark.parametrize(
    "argv,command",
    [
        (["warehouse"], "warehouse"),
        (["info"], "info"),
        (["harvest", "--split", "dev", "--limit", "1"], "harvest"),
        (["label"], "label"),
        (["eval"], "eval"),
        (["loop", "--cycles", "1"], "loop"),
        (["dashboard"], "dashboard"),
    ],
)
def test_parser_dispatch(argv, command):
    ns = build_parser().parse_args(argv)
    assert ns.command == command
    assert callable(ns.func)


def test_cmd_info(capsys):
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert "tier" in out and "hardware" in out


def test_cmd_warehouse(tmp_path):
    path = tmp_path / "w.duckdb"
    assert main(["warehouse", "--path", str(path), "--seed", "1"]) == 0
    assert path.exists()


def test_cmd_dashboard(tmp_path, monkeypatch, capsys):
    from tracetopolicy import dashboard as dash

    target = tmp_path / "d.html"
    monkeypatch.setattr(dash, "render_dashboard", lambda *a, **k: str(target))
    assert main(["dashboard"]) == 0
    assert "dashboard" in capsys.readouterr().out
