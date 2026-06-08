"""auth.py — Password hashing, JWT tokens, user auth helpers. (v5 — improved)"""
from __future__ import annotations
import datetime, os, secrets
from fastapi import Request
from fastapi.responses import RedirectResponse
from jose import jwt, JWTError
from passlib.context import CryptContext
from db import get_conn, now_iso


SECRET_KEY   = os.getenv("SECRET_KEY", "")
ALGORITHM    = "HS256"
TOKEN_EXPIRE = 60 * 24 * 7  # minutes
MIN_PASSWORD_CHARS = 8

# If SECRET_KEY is still empty after app.py's auto-generate, create one
# for this session (won't persist across restarts, but at least the app works)
if not SECRET_KEY:
    import logging
    _log = logging.getLogger("TrendPulse.auth")
    _log.warning(
        "SECRET_KEY not set in environment at import time. "
        "A temporary key is being generated — sessions will NOT survive restart. "
        "app.py should auto-generate and persist a key to .env."
    )
    SECRET_KEY = secrets.token_hex(32)

pwd_ctx = CryptContext(schemes=["pbkdf2_sha256", "bcrypt"], deprecated=["bcrypt"])


def password_validation_error(password: str) -> str:
    """Return a user-safe password validation error, or an empty string when valid."""
    if len(password or "") < MIN_PASSWORD_CHARS:
        return f"Password must be at least {MIN_PASSWORD_CHARS} characters"
    return ""


def hash_password(p: str) -> str:
    error = password_validation_error(p)
    if error:
        raise ValueError(error)
    return pwd_ctx.hash(p)


def verify_password(p: str, h: str) -> bool:
    try:
        return pwd_ctx.verify(p, h)
    except Exception:
        return False


def create_token(uid: str) -> str:
    # Use timezone-aware datetime
    exp = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=TOKEN_EXPIRE)
    return jwt.encode({"sub": uid, "exp": exp}, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str):
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM]).get("sub")
    except JWTError:
        return None


def get_current_user(request: Request):
    token = request.cookies.get("sm_token")
    if not token: return None
    uid = decode_token(token)
    if not uid: return None
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE id=? AND is_active=1",
            (uid,)
        ).fetchone()
    if not row: return None
    user = dict(row)
    # Banned users cannot access the app
    if user.get("is_banned"):
        return None
    return user


def require_user(request: Request):
    """FastAPI dependency — returns the user, or redirects to /login.

    Uses a direct RedirectResponse for HTML navigations and raises 401
    JSON for API/XHR callers (those have `Accept: application/json` or
    an `/api/` path prefix).
    """
    from fastapi import HTTPException
    from fastapi.responses import JSONResponse, RedirectResponse
    user = get_current_user(request)
    if user:
        return user
    accept = (request.headers.get("accept") or "").lower()
    is_api = (
        "application/json" in accept
        or request.url.path.startswith("/api/")
        or request.url.path.startswith("/admin/api/")
    )
    if is_api:
        raise HTTPException(status_code=401, detail="Unauthorized")
    raise HTTPException(
        status_code=307,
        headers={"Location": "/login"},
    )


def require_admin(request: Request):
    """FastAPI dependency — returns admin user or raises 403."""
    from fastapi import HTTPException
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=307, headers={"Location": "/login"})
    if not user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


def update_last_login(uid: str):
    with get_conn() as conn:
        conn.execute("UPDATE users SET last_login=? WHERE id=?", (now_iso(), uid))


def escape_js(s: str) -> str:
    """Safely escape a string for use inside a JS string literal."""
    if not s:
        return ""
    return (
        str(s)
        .replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("<", "\\x3C")
        .replace(">", "\\x3E")
        .replace("&", "\\x26")
    )


def escape_html(s: str) -> str:
    """Basic HTML escape for user-controlled strings in templates."""
    if not s:
        return ""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )
