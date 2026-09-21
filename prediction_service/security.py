"""Password authentication, CSRF helpers, and lightweight login throttling."""

from __future__ import annotations

import secrets
import time
from collections import defaultdict, deque
from collections.abc import MutableMapping

from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import Settings
from .models import AdminUser


PASSWORD_CONTEXT = CryptContext(schemes=["argon2"], deprecated="auto")


def ensure_bootstrap_admin(session: Session, settings: Settings) -> bool:
    """Create the first local administrator only when a password was provided."""

    user = session.scalar(
        select(AdminUser).where(AdminUser.username == settings.admin_username)
    )
    if user is not None:
        return True
    if not settings.admin_password:
        return False
    session.add(
        AdminUser(
            username=settings.admin_username,
            password_hash=PASSWORD_CONTEXT.hash(settings.admin_password),
        )
    )
    return True


def verify_password(session: Session, username: str, password: str) -> bool:
    user = session.scalar(select(AdminUser).where(AdminUser.username == username))
    return bool(user and PASSWORD_CONTEXT.verify(password, user.password_hash))


def csrf_token(session: MutableMapping[str, object]) -> str:
    token = session.get("csrf_token")
    if isinstance(token, str) and token:
        return token
    token = secrets.token_urlsafe(32)
    session["csrf_token"] = token
    return token


def validate_csrf(session: MutableMapping[str, object], supplied: str | None) -> bool:
    expected = session.get("csrf_token")
    return isinstance(expected, str) and bool(supplied) and secrets.compare_digest(expected, supplied)


class LoginRateLimiter:
    """Process-local protection for a single-instance admin panel."""

    def __init__(self, *, max_failures: int = 5, window_seconds: int = 300) -> None:
        self.max_failures = max_failures
        self.window_seconds = window_seconds
        self._attempts: dict[tuple[str, str], deque[float]] = defaultdict(deque)

    def is_allowed(self, client_host: str, username: str) -> bool:
        attempts = self._recent_attempts(client_host, username)
        return len(attempts) < self.max_failures

    def record_failure(self, client_host: str, username: str) -> None:
        self._recent_attempts(client_host, username).append(time.monotonic())

    def clear(self, client_host: str, username: str) -> None:
        self._attempts.pop((client_host, username), None)

    def _recent_attempts(self, client_host: str, username: str) -> deque[float]:
        key = (client_host, username)
        attempts = self._attempts[key]
        cutoff = time.monotonic() - self.window_seconds
        while attempts and attempts[0] < cutoff:
            attempts.popleft()
        return attempts
