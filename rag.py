#!/usr/bin/env python3
"""
Saarland University Module Handbook Assistant — Backward-Compatible CLI Entrypoint.
Delegates to modular production package `app.cli`.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from app.cli import main
except ImportError as exc:
    raise SystemExit(
        "The RAG CLI is incomplete: app/cli.py does not currently provide a "
        "working main() implementation. Restore or implement the application "
        "modules before running this command."
    ) from exc

if __name__ == "__main__":
    main()
