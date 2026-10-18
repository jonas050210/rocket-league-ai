"""Exposure settings shared by ``rlai serve`` and the FastAPI app.

The point of ``AppSettings`` is that the security defaults live in exactly one
validated place: a loopback bind, no CORS, and an explicit token.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from rlai import cli
from rlai.api import create_app
from rlai.config import AppSettings


def test_defaults_are_local_only():
    settings = AppSettings()
    assert settings.host == "127.0.0.1"
    assert settings.port == 8000
    assert settings.auth_token is None
    assert settings.cors_origins == []
    assert settings.is_public is False
    assert settings.exposed_without_token is False
    assert settings.weak_token is False


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "rlpi.local", "::"])
def test_non_loopback_hosts_count_as_public(host):
    settings = AppSettings(host=host)
    assert settings.is_public is True
    assert settings.exposed_without_token is True
    # with a token the exposure is deliberate, not accidental
    assert AppSettings(host=host, auth_token="a-long-enough-token").exposed_without_token is False


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_loopback_hosts_are_not_public(host):
    assert AppSettings(host=host).is_public is False


@pytest.mark.parametrize("host", ["", "http://localhost", "0.0.0.0/0", "a b", "host?x=1"])
def test_host_must_be_an_interface_not_a_url(host):
    with pytest.raises(ValidationError, match="host"):
        AppSettings(host=host)


@pytest.mark.parametrize("port", [0, -1, 65536, 100000])
def test_port_bounds(port):
    with pytest.raises(ValidationError):
        AppSettings(port=port)
    assert AppSettings(port=1).port == 1 and AppSettings(port=65535).port == 65535


def test_cors_origins_are_explicit():
    settings = AppSettings(cors_origins=["https://gui.example/", "http://127.0.0.1:3000"])
    assert settings.cors_origins == ["https://gui.example", "http://127.0.0.1:3000"]
    assert AppSettings(cors_origins=["", "  "]).cors_origins == []

    with pytest.raises(ValidationError, match=r"\*"):
        AppSettings(cors_origins=["*"])
    with pytest.raises(ValidationError, match="http"):
        AppSettings(cors_origins=["gui.example"])


def test_weak_token_is_reported_not_fatal():
    assert AppSettings(auth_token="short").weak_token is True
    assert AppSettings(auth_token="a-long-enough-token").weak_token is False
    # an empty token means "local only", not "weak"
    assert AppSettings(auth_token="   ").auth_token is None
    assert AppSettings(auth_token="   ").weak_token is False


def test_from_env_and_overrides(monkeypatch):
    monkeypatch.setenv("RLAI_HOST", "0.0.0.0")
    monkeypatch.setenv("RLAI_PORT", "9001")
    monkeypatch.setenv("RLAI_AUTH_TOKEN", "environment-token")
    monkeypatch.setenv("RLAI_CORS_ORIGINS", "https://a.example, https://b.example/")

    settings = AppSettings.from_env()
    assert settings.host == "0.0.0.0"
    assert settings.port == 9001
    assert settings.auth_token == "environment-token"
    assert settings.cors_origins == ["https://a.example", "https://b.example"]
    assert settings.export_env() == {
        "RLAI_AUTH_TOKEN": "environment-token",
        "RLAI_CORS_ORIGINS": "https://a.example,https://b.example",
    }

    # an explicit CLI argument wins over the environment
    override = AppSettings.from_env(host="127.0.0.1", port=8000)
    assert override.host == "127.0.0.1" and override.port == 8000
    assert override.auth_token == "environment-token"

    monkeypatch.delenv("RLAI_HOST")
    monkeypatch.delenv("RLAI_PORT")
    monkeypatch.delenv("RLAI_AUTH_TOKEN")
    monkeypatch.delenv("RLAI_CORS_ORIGINS")
    assert AppSettings.from_env() == AppSettings()


def test_serve_binds_loopback_and_warns_on_exposure(monkeypatch, capsys, data_root):
    seen = {}

    class _Uvicorn:
        @staticmethod
        def run(app, host, port, log_level):
            seen.update(host=host, port=port, log_level=log_level)

    monkeypatch.setitem(__import__("sys").modules, "uvicorn", _Uvicorn)
    monkeypatch.setattr("rlai.api.create_app", lambda: object())

    assert cli.main(["serve"]) == 0
    assert seen["host"] == "127.0.0.1" and seen["port"] == 8000
    assert "WARNING" not in capsys.readouterr().err

    assert cli.main(["serve", "--host", "0.0.0.0", "--port", "9000"]) == 0
    assert seen["host"] == "0.0.0.0" and seen["port"] == 9000
    error = capsys.readouterr().err
    assert "WARNING: serving on 0.0.0.0 without a token" in error
    assert "127.0.0.1" in error

    # a token silences the exposure warning, a short one adds its own
    assert cli.main(["serve", "--host", "0.0.0.0", "--token", "a-long-enough-token"]) == 0
    assert "WARNING" not in capsys.readouterr().err
    assert cli.main(["serve", "--token", "short"]) == 0
    assert "shorter than 12 characters" in capsys.readouterr().err


def test_serve_rejects_a_url_as_host(monkeypatch, capsys, data_root):
    with pytest.raises(SystemExit):
        # argparse accepts the value; AppSettings must reject it before binding
        cli.main(["serve", "--host", "http://0.0.0.0"])
    assert "host must be an interface" in capsys.readouterr().err


def test_api_uses_the_same_settings(monkeypatch, data_root, models_dir):
    monkeypatch.setenv("RLAI_CORS_ORIGINS", "https://gui.example")
    monkeypatch.setenv("RLAI_AUTH_TOKEN", "short")
    app = create_app()
    with TestClient(app) as client:
        settings = app.state.settings
        assert settings.cors_origins == ["https://gui.example"]
        assert settings.weak_token is True
        assert client.get("/api/health").status_code == 200
        # mutations need the token, and the weak token was logged, not ignored
        assert client.post("/api/config/validate", json={}).status_code == 401
        assert client.post(
            "/api/config/validate", json={}, headers={"x-rlai-token": "short"}
        ).status_code == 200
        assert client.post(
            "/api/config/validate", json={"mode": "hoops"}, headers={"x-rlai-token": "short"}
        ).status_code == 422
        from rlai.store import Store

        store = Store(data_root / "rlai.sqlite3")
        assert any(
            "shorter than 12 characters" in row["message"]
            for row in store.tail_logs(None, 0, 50)
        )
