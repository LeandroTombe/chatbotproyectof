"""
REST API server entry point.

Run with:
    python server.py

Or directly with uvicorn:
    uvicorn server:app --host 0.0.0.0 --port 9000 --reload
"""
from __future__ import annotations

import logging
import sys

import uvicorn

from api.app import create_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)

# Module-level `app` so uvicorn can reference it as "server:app"
app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=9000,
        reload=False,   # set True during development
        log_level="info",
    )
