"""Tests for dashboard configuration loading."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from bot.dashboard.config import load_dashboard_config

if TYPE_CHECKING:
    import pytest


class TestDashboardConfig:
    """Test dashboard config parsing from environment."""

    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Dashboard disabled by default."""
        # Clear all dashboard env vars
        for key in os.environ:
            if key.startswith("DASHBOARD_"):
                monkeypatch.delenv(key, raising=False)
        monkeypatch.delenv("OWNER_IDS", raising=False)

        cfg = load_dashboard_config()
        assert cfg.enabled is False
        assert cfg.host == "127.0.0.1"
        assert cfg.port == 8011
        assert cfg.public_bind is False
        assert cfg.auth_token is None
        assert cfg.owner_ids == set()

    def test_enabled_with_auth_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Dashboard enabled when flag is true and auth token provided."""
        monkeypatch.setenv("DASHBOARD_ENABLED", "true")
        monkeypatch.setenv("DASHBOARD_AUTH_TOKEN", "test-secret-123")
        monkeypatch.delenv("DASHBOARD_SESSION_SECRET", raising=False)
        monkeypatch.delenv("OWNER_IDS", raising=False)

        cfg = load_dashboard_config()
        assert cfg.enabled is True
        assert cfg.auth_token == "test-secret-123"

    def test_enabled_without_auth_token_fails_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Dashboard disabled if enabled=true but no auth token."""
        monkeypatch.setenv("DASHBOARD_ENABLED", "true")
        monkeypatch.delenv("DASHBOARD_AUTH_TOKEN", raising=False)
        monkeypatch.delenv("DASHBOARD_SESSION_SECRET", raising=False)
        monkeypatch.delenv("OWNER_IDS", raising=False)

        cfg = load_dashboard_config()
        assert cfg.enabled is False  # Fails closed

    def test_custom_host_port(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Custom host and port from env."""
        monkeypatch.setenv("DASHBOARD_ENABLED", "true")
        monkeypatch.setenv("DASHBOARD_AUTH_TOKEN", "test-secret")
        monkeypatch.setenv("DASHBOARD_HOST", "0.0.0.0")
        monkeypatch.setenv("DASHBOARD_PORT", "9000")
        monkeypatch.delenv("DASHBOARD_SESSION_SECRET", raising=False)
        monkeypatch.delenv("OWNER_IDS", raising=False)

        cfg = load_dashboard_config()
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 9000

    def test_public_bind(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Public bind flag."""
        monkeypatch.setenv("DASHBOARD_ENABLED", "true")
        monkeypatch.setenv("DASHBOARD_AUTH_TOKEN", "test-secret")
        monkeypatch.setenv("DASHBOARD_PUBLIC_BIND", "true")
        monkeypatch.delenv("DASHBOARD_SESSION_SECRET", raising=False)
        monkeypatch.delenv("OWNER_IDS", raising=False)

        cfg = load_dashboard_config()
        assert cfg.public_bind is True

    def test_owner_ids_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Owner IDs parsed from comma-separated string."""
        monkeypatch.setenv("DASHBOARD_ENABLED", "true")
        monkeypatch.setenv("DASHBOARD_AUTH_TOKEN", "test-secret")
        monkeypatch.setenv("DASHBOARD_OWNER_IDS", "123,456,789")
        monkeypatch.delenv("DASHBOARD_SESSION_SECRET", raising=False)
        monkeypatch.delenv("OWNER_IDS", raising=False)

        cfg = load_dashboard_config()
        assert cfg.owner_ids == {123, 456, 789}

    def test_owner_ids_merged_with_existing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Dashboard owner IDs merged with global OWNER_IDS."""
        monkeypatch.setenv("DASHBOARD_ENABLED", "true")
        monkeypatch.setenv("DASHBOARD_AUTH_TOKEN", "test-secret")
        monkeypatch.setenv("DASHBOARD_OWNER_IDS", "100")
        monkeypatch.setenv("OWNER_IDS", "200,300")
        monkeypatch.delenv("DASHBOARD_SESSION_SECRET", raising=False)

        cfg = load_dashboard_config()
        assert cfg.owner_ids == {100, 200, 300}

    def test_rate_limit_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Rate limit configuration."""
        monkeypatch.setenv("DASHBOARD_ENABLED", "true")
        monkeypatch.setenv("DASHBOARD_AUTH_TOKEN", "test-secret")
        monkeypatch.setenv("DASHBOARD_RATE_LIMIT_SENDS_PER_MINUTE", "10")
        monkeypatch.delenv("DASHBOARD_SESSION_SECRET", raising=False)
        monkeypatch.delenv("OWNER_IDS", raising=False)

        cfg = load_dashboard_config()
        assert cfg.rate_limit_sends_per_minute == 10

    def test_retention_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Retention configuration."""
        monkeypatch.setenv("DASHBOARD_ENABLED", "true")
        monkeypatch.setenv("DASHBOARD_AUTH_TOKEN", "test-secret")
        monkeypatch.setenv("DASHBOARD_DM_RETENTION_DAYS", "30")
        monkeypatch.setenv("DASHBOARD_AUDIT_RETENTION_DAYS", "365")
        monkeypatch.delenv("DASHBOARD_SESSION_SECRET", raising=False)
        monkeypatch.delenv("OWNER_IDS", raising=False)

        cfg = load_dashboard_config()
        assert cfg.dm_retention_days == 30
        assert cfg.audit_retention_days == 365

    def test_invalid_port_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Invalid port falls back to default."""
        monkeypatch.setenv("DASHBOARD_ENABLED", "true")
        monkeypatch.setenv("DASHBOARD_AUTH_TOKEN", "test-secret")
        monkeypatch.setenv("DASHBOARD_PORT", "not-a-number")
        monkeypatch.delenv("DASHBOARD_SESSION_SECRET", raising=False)
        monkeypatch.delenv("OWNER_IDS", raising=False)

        cfg = load_dashboard_config()
        assert cfg.port == 8011  # Default

    def test_disabled_means_no_start(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When disabled, no server should start."""
        monkeypatch.setenv("DASHBOARD_ENABLED", "false")
        monkeypatch.delenv("DASHBOARD_AUTH_TOKEN", raising=False)
        monkeypatch.delenv("OWNER_IDS", raising=False)

        cfg = load_dashboard_config()
        assert cfg.enabled is False
