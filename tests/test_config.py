"""Tests for application configuration module."""

from __future__ import annotations

from app.config import Settings, get_settings


class TestSettings:
    def test_default_database_url(self):
        s = Settings()
        assert "sqlite" in s.database_url or "postgresql" in s.database_url

    def test_default_port(self):
        s = Settings()
        assert s.api_port == 8000

    def test_default_rate_limit(self):
        s = Settings()
        assert s.rate_limit == 300

    def test_default_service_level(self):
        s = Settings()
        assert s.default_service_level == 0.95

    def test_repr_contains_host(self):
        s = Settings()
        assert "host=" in repr(s)

    def test_env_override_rate_limit(self, monkeypatch):
        monkeypatch.setenv("RATE_LIMIT", "100")
        s = Settings()
        assert s.rate_limit == 100

    def test_env_override_log_level(self, monkeypatch):
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        s = Settings()
        assert s.log_level == "DEBUG"


class TestGetSettings:
    def test_returns_settings_instance(self):
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_cached_singleton(self):
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2
