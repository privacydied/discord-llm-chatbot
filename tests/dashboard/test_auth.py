"""Tests for dashboard authentication and authorization."""

from __future__ import annotations

from unittest.mock import MagicMock

from bot.dashboard.auth import SessionStore, _check_bearer_auth, _check_session_auth


class TestSessionStore:
    """Test in-memory session store."""

    def test_create_and_get(self) -> None:
        store = SessionStore(session_ttl_hours=8)
        csrf = "test-csrf"
        session_id = store.create(user_id=123, csrf_token=csrf)
        assert session_id is not None

        session = store.get(session_id)
        assert session is not None
        assert session["user_id"] == 123
        assert session["csrf_token"] == csrf

    def test_remove(self) -> None:
        store = SessionStore()
        sid = store.create(user_id=123, csrf_token="token")
        store.remove(sid)
        assert store.get(sid) is None

    def test_expiry(self) -> None:
        store = SessionStore(session_ttl_hours=0)  # Immediate expiry
        sid = store.create(user_id=123, csrf_token="token")
        import time

        time.sleep(0.01)  # Tiny delay
        assert store.get(sid) is None

    def test_cleanup(self) -> None:
        store = SessionStore(session_ttl_hours=0)
        store.create(user_id=1, csrf_token="t1")
        store.create(user_id=2, csrf_token="t2")
        import time

        time.sleep(0.01)
        removed = store.cleanup()
        assert removed == 2


class TestBearerAuth:
    """Test bearer token authentication."""

    def test_valid_bearer(self) -> None:
        request = MagicMock()
        request.headers = {"Authorization": "Bearer my-secret-token"}
        assert _check_bearer_auth(request, "my-secret-token") is True

    def test_invalid_bearer(self) -> None:
        request = MagicMock()
        request.headers = {"Authorization": "Bearer wrong-token"}
        assert _check_bearer_auth(request, "my-secret-token") is False

    def test_no_auth_header(self) -> None:
        request = MagicMock()
        request.headers = {}
        assert _check_bearer_auth(request, "my-secret-token") is False

    def test_basic_auth_not_bearer(self) -> None:
        request = MagicMock()
        request.headers = {"Authorization": "Basic abc123"}
        assert _check_bearer_auth(request, "my-secret-token") is False


class TestSessionAuth:
    """Test session cookie authentication."""

    def test_valid_session(self) -> None:
        store = SessionStore()
        sid = store.create(user_id=123, csrf_token="csrf123")

        request = MagicMock()
        request.cookies = {"dash_session": sid}

        session = _check_session_auth(request, store)
        assert session is not None
        assert session["user_id"] == 123

    def test_invalid_session(self) -> None:
        store = SessionStore()
        request = MagicMock()
        request.cookies = {"dash_session": "nonexistent"}
        assert _check_session_auth(request, store) is None

    def test_no_cookie(self) -> None:
        store = SessionStore()
        request = MagicMock()
        request.cookies = {}
        assert _check_session_auth(request, store) is None
