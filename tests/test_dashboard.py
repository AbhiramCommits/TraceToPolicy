from tracetopolicy.dashboard import render_dashboard


def test_render_dashboard_creates_self_contained_html(tmp_path, monkeypatch):
    from tracetopolicy import dashboard as dash

    monkeypatch.setattr(dash, "RESULTS", "results")
    monkeypatch.setattr(dash, "REGISTRY", "artifacts/registry.json")
    out = tmp_path / "dashboard.html"
    path = render_dashboard(str(out))
    html = out.read_text()
    assert path == str(out)
    assert "TraceToPolicy" in html
    assert "<table" in html
    # self-contained: no external stylesheet/script references
    assert "http://" not in html and "https://" not in html
