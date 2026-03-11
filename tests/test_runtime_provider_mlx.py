from hermes_cli import runtime_provider as rp


def test_resolve_runtime_provider_mlx(monkeypatch):
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "mlx")
    resolved = rp.resolve_runtime_provider(requested="mlx")

    assert resolved["provider"] == "mlx"
    assert resolved["api_mode"] == "mlx_local"
    assert resolved["base_url"] == "mlx://local"
    assert resolved["api_key"] == "mlx-local"
