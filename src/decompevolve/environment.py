"""Load project-local credentials without replacing explicit shell settings."""
from decompevolve.data import ROOT


def load_environment(path=ROOT / ".env"):
    from dotenv import load_dotenv

    # Parse as data, never execute shell commands or expand variable references.
    load_dotenv(dotenv_path=path, override=False, interpolate=False, encoding="utf-8-sig")
