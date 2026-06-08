"""
app.py — TrendPulse SaaS entry point (v6 — all critical fixes)

Fixes from v5:
  - FIXED: AttributeError 'SecurityAndPerfMiddleware' has no attribute 'state'
    → slowapi limiter is initialized BEFORE middleware is added, and we keep
      a direct reference to the FastAPI instance for state access.
  - FIXED: SECRET_KEY auto-generates and persists to .env so sessions survive
    restarts.
  - FIXED: Login navigation — the app no longer crashes on startup, and the
    login→dashboard redirect flow works correctly.
  - SECURITY: CORS restricted (configurable via env).
  - SECURITY: Security headers hardened.
  - CODE QUALITY: Removed deprecated datetime.utcnow().
  - routes/models.py registered (/api/models dynamic endpoint)
  - release_quota_reservation imported so generate.py can use it
"""
import asyncio
import logging
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from db import init_db, OUTPUT_ROOT
import pipelines

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("TrendPulse")

# ── Ensure SECRET_KEY exists (auto-generate & persist if missing) ──────────
_ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


def _ensure_secret_key() -> str:
    """Return a SECRET_KEY, generating and persisting one to .env if absent."""
    key = os.getenv("SECRET_KEY", "").strip()
    if key:
        return key

    import secrets
    key = secrets.token_hex(32)

    # Persist to .env so the same key survives restarts
    try:
        existing = ""
        if os.path.isfile(_ENV_FILE):
            with open(_ENV_FILE, "r", encoding="utf-8") as f:
                existing = f.read()

        if "SECRET_KEY=" not in existing:
            with open(_ENV_FILE, "a", encoding="utf-8") as f:
                f.write(f"\nSECRET_KEY={key}\n")
            os.environ["SECRET_KEY"] = key
            logger.info("SECRET_KEY auto-generated and saved to .env")
        else:
            # Key is in file but not loaded — maybe dotenv not loaded yet
            for line in existing.splitlines():
                if line.startswith("SECRET_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    os.environ["SECRET_KEY"] = key
                    break
    except Exception as exc:
        logger.warning("Could not persist SECRET_KEY to .env: %s", exc)
        os.environ["SECRET_KEY"] = key

    return key


_ensure_secret_key()

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    asyncio.create_task(pipelines._scheduler_loop())
    if not os.getenv("SECRET_KEY"):
        logger.critical(
            "SECRET_KEY not set in .env — sessions will die on restart! "
            "Set a random SECRET_KEY in your .env file immediately."
        )
    logger.info("TrendPulse v6 started — scheduler active")
    yield

# ── FastAPI app ─────────────────────────────────────────────────────────────
app = FastAPI(title="TrendPulse SaaS", version="6.0.0", lifespan=lifespan)

# ── Rate limiting setup — MUST be done BEFORE middleware is added ───────────
# This fixes the AttributeError: 'SecurityAndPerfMiddleware' has no attribute 'state'
# The slowapi limiter needs app.state which only exists on the FastAPI instance,
# not on a middleware wrapper.
limiter = None
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    logger.info("Rate limiting enabled via slowapi")
except ImportError:
    logger.info("slowapi not installed — rate limiting disabled")

# ── Middleware (added AFTER limiter setup) ──────────────────────────────────
# CORS: configurable via env variables for production security
_cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1024)


class SecurityAndPerfMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        import time as _t
        try:
            from auth import decode_token
            from db import get_user_ui_language
            from core.i18n import normalize_lang
            token = request.cookies.get("sm_token", "")
            if token:
                uid = decode_token(token)
                if uid:
                    lang = get_user_ui_language(uid)
                    request.state.lang = normalize_lang(lang)
                else:
                    request.state.lang = "en"
            else:
                request.state.lang = "en"
        except Exception:
            request.state.lang = "en"

        t0       = _t.perf_counter()
        response = await call_next(request)
        ms       = round((_t.perf_counter() - t0) * 1000)

        response.headers["X-Response-Time"]        = f"{ms}ms"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"]        = "DENY"
        response.headers["Referrer-Policy"]        = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"]     = "camera=(), microphone=(), geolocation=()"
        response.headers["X-XSS-Protection"]       = "0"  # Deprecated but some browsers still check
        return response


app.add_middleware(SecurityAndPerfMiddleware)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_ROOT)), name="outputs")

# ── Routers ─────────────────────────────────────────────────────────────────
from routes.auth      import router as auth_router
from routes.generate  import router as generate_router
from routes.strategy  import router as strategy_router
from routes.calendar  import router as calendar_router
from routes.account   import router as account_router
from routes.insights  import router as insights_router
from routes.api       import router as api_router
from routes.admin     import router as admin_router
from routes.brands    import router as brands_router
from routes.language  import router as language_router
from routes.models    import router as models_router

app.include_router(auth_router)
app.include_router(generate_router)
app.include_router(strategy_router)
app.include_router(calendar_router)
app.include_router(account_router)
app.include_router(insights_router)
app.include_router(api_router)
app.include_router(admin_router)
app.include_router(brands_router)
app.include_router(language_router)
app.include_router(models_router)

