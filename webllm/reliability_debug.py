"""Server-side bridge for the reliability debug API (Phase 2.1.3).

Developer feature, OFF by default. Enablement is configuration-driven; the
web layer's configuration source is the environment:

    DOZEN_RELIABILITY_DEBUG=1          enable the debug endpoints
    DOZEN_RELIABILITY_DEBUG_LOG=1      also log one line per recorded attempt
    DOZEN_RELIABILITY_DEBUG_MAX=100    max attempts per response

The bridge builds one ReliabilityDebugApi over the process-wide default
recorder (the one build_orchestrator's decorator records into). Read-only.
"""

from __future__ import annotations

import os
import threading
from typing import Optional

from dozen.reliability.config import ObservabilityConfig
from dozen.reliability.debug import ReliabilityDebugApi
from dozen.reliability.recorder import default_recorder

_ENV_ENABLE = "DOZEN_RELIABILITY_DEBUG"
_ENV_LOG = "DOZEN_RELIABILITY_DEBUG_LOG"
_ENV_MAX = "DOZEN_RELIABILITY_DEBUG_MAX"
_TRUTHY = {"1", "true", "yes", "on"}

_api: Optional[ReliabilityDebugApi] = None
_lock = threading.Lock()


def _config_from_env() -> ObservabilityConfig:
    try:
        max_returned = int(os.environ.get(_ENV_MAX, "100"))
    except ValueError:
        max_returned = 100
    return ObservabilityConfig(
        enable_debug_api=os.environ.get(_ENV_ENABLE, "").strip().lower() in _TRUTHY,
        enable_attempt_logging=os.environ.get(_ENV_LOG, "").strip().lower() in _TRUTHY,
        max_attempts_returned=max(1, max_returned),
        statistics_cache_seconds=2.0,
    )


def get_debug_api() -> ReliabilityDebugApi:
    """Singleton facade; its ``enabled`` flag decides endpoint availability."""
    global _api
    with _lock:
        if _api is None:
            _api = ReliabilityDebugApi(default_recorder(), _config_from_env())
        return _api


def reset_debug_api() -> None:
    """Drop the singleton so tests can re-read the environment."""
    global _api
    with _lock:
        _api = None
