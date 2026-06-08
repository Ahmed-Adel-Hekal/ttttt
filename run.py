"""run.py — Development / production entry point for TrendPulse."""
import os
import sys
import logging

logger = logging.getLogger("TrendPulse.startup")


def check_environment():
    """Pre-flight environment checks."""
    warnings = []
    errors   = []

    # DB path
    db_path = os.getenv("DATABASE_PATH", "data/saas.db")
    db_dir  = os.path.dirname(db_path) or "."
    if not os.path.isdir(db_dir):
        try:
            os.makedirs(db_dir, exist_ok=True)
        except OSError as e:
            errors.append(f"Cannot create database directory {db_dir}: {e}")

    # Output dir
    out_dir = os.getenv("OUTPUT_DIR", "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # Processed dir
    proc_dir = "data/processed"
    os.makedirs(proc_dir, exist_ok=True)

    # SECRET_KEY — warn if still missing (app.py auto-generates but warn anyway)
    if not os.getenv("SECRET_KEY"):
        warnings.append(
            "SECRET_KEY is not set in .env — app.py will auto-generate one, "
            "but for production, set SECRET_KEY explicitly in your .env file."
        )

    for w in warnings:
        print(f"\033[33m{w}\033[0m", file=sys.stderr)

    if errors:
        for e in errors:
            print(f"\033[31mFATAL: {e}\033[0m", file=sys.stderr)
        sys.exit(1)

    return len(warnings) == 0


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    check_environment()

    import uvicorn
    host  = os.getenv("HOST", "0.0.0.0")
    port  = int(os.getenv("PORT", "8000"))
    debug = os.getenv("DEBUG", "false").lower() == "true"

    print(f"\n\033[36m  TrendPulse v6  ->  http://{host}:{port}\033[0m\n")

    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info",
        access_log=True,
    )
