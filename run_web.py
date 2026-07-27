"""Entrypoint for the Dozen web-automation orchestrator.

    python run_web.py            # http://127.0.0.1:8000

First run only:
    pip install -r requirements.txt
    python -m playwright install chromium
"""

from __future__ import annotations

import uvicorn

if __name__ == "__main__":
    # reload=False is important: the BrowserManager owns OS browser windows and
    # a worker thread that must not be torn down by the autoreloader.
    uvicorn.run("webllm.server:app", host="127.0.0.1", port=8000, reload=False)
