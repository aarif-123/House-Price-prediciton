"""
Vercel entrypoint.

Vercel's Python runtime looks for an ASGI app named `app` inside `api/*.py`.
We re-export the FastAPI app defined in `main.py`.
"""

from main import app  # noqa: F401

