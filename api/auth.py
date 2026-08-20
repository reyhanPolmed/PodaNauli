"""Local stakeholder authentication backed by the runtime SQLite database."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256, scrypt
import hmac
import os
from pathlib import Path
import secrets
import sqlite3
from threading import RLock
import time


SESSION_COOKIE = "podanauli_session"
MAX_LOGIN_ATTEMPTS = 5
LOGIN_WINDOW_SECONDS = 5 * 60


class AuthenticationError(ValueError):
    """Raised when credentials or an authenticated session are invalid."""


class AuthenticationUnavailableError(RuntimeError):
    """Raised when no admin password has been configured."""


class LoginRateLimitError(RuntimeError):
    """Raised after repeated failed login attempts."""


@dataclass(frozen=True)
class AuthenticatedUser:
    id: int
    username: str
    display_name: str
    role: str

    def payload(self) -> dict[str, str]:
        return {
            "username": self.username,
            "display_name": self.display_name,
            "role": self.role,
        }


class AuthService:
    """Manage one stakeholder admin account and opaque server-side sessions."""

    def __init__(self, runtime_dir: Path) -> None:
        runtime_dir.mkdir(parents=True, exist_ok=True)
        self.database_path = runtime_dir / "auth.sqlite3"
        self.session_hours = max(1, min(int(os.getenv("PODANAULI_SESSION_HOURS", "8")), 24))
        self.cookie_secure = os.getenv("PODANAULI_COOKIE_SECURE", "false").casefold() == "true"
        self._lock = RLock()
        self._failed_logins: dict[str, list[float]] = {}
        self._initialize_database()
        self._configure_admin()

    @property
    def configured(self) -> bool:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS total FROM users WHERE active = 1").fetchone()
        return bool(row["total"])

    def login(
        self,
        username: str,
        password: str,
        *,
        client_address: str,
        user_agent: str,
    ) -> tuple[AuthenticatedUser, str]:
        if not self.configured:
            raise AuthenticationUnavailableError(
                "Akun stakeholder belum dikonfigurasi pada environment server."
            )

        key = self._attempt_key(username, client_address)
        self._enforce_rate_limit(key)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, username, display_name, role, password_hash "
                "FROM users WHERE username = ? COLLATE NOCASE AND active = 1",
                (username.strip(),),
            ).fetchone()

        if row is None or not self._verify_password(password, row["password_hash"]):
            self._record_failed_login(key)
            self.audit("login_failed", None, client_address=client_address)
            raise AuthenticationError("Nama pengguna atau password tidak sesuai.")

        self._failed_logins.pop(key, None)
        user = AuthenticatedUser(
            id=int(row["id"]),
            username=str(row["username"]),
            display_name=str(row["display_name"]),
            role=str(row["role"]),
        )
        token = secrets.token_urlsafe(48)
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(hours=self.session_hours)
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM sessions WHERE expires_at <= ?", (now.timestamp(),))
            connection.execute(
                "INSERT INTO sessions "
                "(user_id, token_hash, created_at, expires_at, client_hash, user_agent) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    user.id,
                    self._token_hash(token),
                    now.timestamp(),
                    expires_at.timestamp(),
                    self._privacy_hash(client_address),
                    user_agent[:300],
                ),
            )
            connection.commit()
        self.audit("login_success", user, client_address=client_address)
        return user, token

    def authenticate(self, token: str | None) -> AuthenticatedUser | None:
        if not token:
            return None
        now = datetime.now(timezone.utc).timestamp()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT users.id, users.username, users.display_name, users.role "
                "FROM sessions JOIN users ON users.id = sessions.user_id "
                "WHERE sessions.token_hash = ? AND sessions.expires_at > ? AND users.active = 1",
                (self._token_hash(token), now),
            ).fetchone()
        if row is None:
            return None
        return AuthenticatedUser(
            id=int(row["id"]),
            username=str(row["username"]),
            display_name=str(row["display_name"]),
            role=str(row["role"]),
        )

    def logout(self, token: str | None, user: AuthenticatedUser | None = None) -> None:
        if token:
            with self._lock, self._connect() as connection:
                connection.execute("DELETE FROM sessions WHERE token_hash = ?", (self._token_hash(token),))
                connection.commit()
        if user:
            self.audit("logout", user)

    def audit(
        self,
        action: str,
        user: AuthenticatedUser | None,
        *,
        target: str | None = None,
        client_address: str = "",
    ) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO audit_log (user_id, username, action, target, client_hash, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    user.id if user else None,
                    user.username if user else None,
                    action,
                    target,
                    self._privacy_hash(client_address) if client_address else None,
                    datetime.now(timezone.utc).timestamp(),
                ),
            )
            connection.commit()

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    display_name TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role = 'admin'),
                    password_hash TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    token_hash TEXT NOT NULL UNIQUE,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    client_hash TEXT,
                    user_agent TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at);
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    username TEXT,
                    action TEXT NOT NULL,
                    target TEXT,
                    client_hash TEXT,
                    created_at REAL NOT NULL
                );
                """
            )
            connection.commit()

    def _configure_admin(self) -> None:
        password = os.getenv("PODANAULI_ADMIN_PASSWORD", "")
        if not password:
            return
        if len(password) < 12:
            raise ValueError("PODANAULI_ADMIN_PASSWORD harus terdiri dari minimal 12 karakter.")
        username = os.getenv("PODANAULI_ADMIN_USERNAME", "stakeholder").strip() or "stakeholder"
        now = datetime.now(timezone.utc).timestamp()
        password_hash = self._hash_password(password)
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO users (username, display_name, role, password_hash, active, created_at, updated_at)
                VALUES (?, 'Stakeholder', 'admin', ?, 1, ?, ?)
                ON CONFLICT(username) DO UPDATE SET
                    display_name = 'Stakeholder',
                    role = 'admin',
                    password_hash = excluded.password_hash,
                    active = 1,
                    updated_at = excluded.updated_at
                """,
                (username, password_hash, now, now),
            )
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @staticmethod
    def _hash_password(password: str) -> str:
        salt = secrets.token_bytes(16)
        digest = scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=64)
        return f"scrypt$16384$8$1${salt.hex()}${digest.hex()}"

    @staticmethod
    def _verify_password(password: str, encoded: str) -> bool:
        try:
            algorithm, n, r, p, salt, expected = encoded.split("$", 5)
            if algorithm != "scrypt":
                return False
            actual = scrypt(
                password.encode("utf-8"),
                salt=bytes.fromhex(salt),
                n=int(n),
                r=int(r),
                p=int(p),
                dklen=len(bytes.fromhex(expected)),
            )
            return hmac.compare_digest(actual, bytes.fromhex(expected))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _token_hash(token: str) -> str:
        return sha256(token.encode("utf-8")).hexdigest()

    @staticmethod
    def _privacy_hash(value: str) -> str:
        return sha256(value.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _attempt_key(username: str, client_address: str) -> str:
        return sha256(f"{username.casefold()}|{client_address}".encode("utf-8")).hexdigest()

    def _enforce_rate_limit(self, key: str) -> None:
        now = time.monotonic()
        attempts = [value for value in self._failed_logins.get(key, []) if now - value < LOGIN_WINDOW_SECONDS]
        self._failed_logins[key] = attempts
        if len(attempts) >= MAX_LOGIN_ATTEMPTS:
            raise LoginRateLimitError("Terlalu banyak percobaan login. Coba kembali dalam beberapa menit.")

    def _record_failed_login(self, key: str) -> None:
        self._failed_logins.setdefault(key, []).append(time.monotonic())
