"""Runtime build identification (live-acceptance correction).

WHY THIS EXISTS
---------------
A live "give code only" run returned architecture prose and refusals — a
failure the audited Phase 4F tree does not produce. The most dangerous
possibility is the least visible one: a server process quietly importing a
DIFFERENT, older workspace than the one under audit, so a fix is "verified"
against code that never ran.

This module makes the running build self-identifying. It is pure and
content-free: it reports WHICH source tree answered a request and a
deterministic fingerprint of the delivery-critical logic, never any request,
prompt, conversation or provider text. A stale checkout produces a different
fingerprint, so it can never again be tested unknowingly.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .finalization import (
    FINALIZATION_SCHEMA_VERSION,
    ORDERED_SECTION_SCHEMA_VERSION,
    PROSE_ENVELOPE_SCHEMA_VERSION,
)

# This tree contains the live-acceptance delta and is under independent audit;
# calling it the already-audited baseline would erase the distinction the
# runtime identity exists to expose.
TEST_BASELINE_LABEL = "phase4f-live-fix-candidate"

# The delivery-critical modules whose bytes define the runtime behavior that
# matters for the code-only contract. If any changes, the fingerprint changes.
_FINGERPRINT_SOURCES = (
    "dozen/intent.py",
    "dozen/planner.py",
    "dozen/orchestrator.py",
    "dozen/finalization.py",
    "dozen/artifact_results.py",
    "dozen/models.py",
    "dozen/build_info.py",
    "webllm/server.py",
    "webllm/static/index.html",
)

_PACKAGE_DIR = Path(__file__).resolve().parent
_SOURCE_ROOT = _PACKAGE_DIR.parent


def _schema_signature() -> str:
    return (
        f"fin={FINALIZATION_SCHEMA_VERSION};"
        f"prose={PROSE_ENVELOPE_SCHEMA_VERSION};"
        f"section={ORDERED_SECTION_SCHEMA_VERSION};"
        f"baseline={TEST_BASELINE_LABEL}"
    )


def build_fingerprint() -> str:
    """A deterministic 16-hex-char fingerprint of the delivery-critical build.

    Derived from the finalization schema versions plus the exact bytes of the
    modules that decide the code-only contract. Content-free: it hashes source,
    never user data, so it is stable across runs of the same tree and different
    across trees.
    """
    digest = hashlib.sha256()
    digest.update(_schema_signature().encode("utf-8"))
    for name in _FINGERPRINT_SOURCES:
        digest.update(b"\x00" + name.encode("utf-8") + b"\x00")
        try:
            digest.update((_SOURCE_ROOT / name).read_bytes())
        except OSError:
            digest.update(b"<unreadable>")
    return digest.hexdigest()[:16]


def build_identity(*, reveal_paths: bool = False) -> dict[str, Any]:
    """The runtime build identity.

    ``reveal_paths=False`` (the default for any remotely reachable surface)
    omits absolute filesystem paths; the fingerprint, schema version and
    baseline label remain, which is enough to tell two trees apart without
    disclosing the server's layout to ordinary users.
    """
    identity: dict[str, Any] = {
        "build_fingerprint": build_fingerprint(),
        "finalization_schema_version": FINALIZATION_SCHEMA_VERSION,
        "prose_envelope_schema_version": PROSE_ENVELOPE_SCHEMA_VERSION,
        "ordered_section_schema_version": ORDERED_SECTION_SCHEMA_VERSION,
        "test_baseline": TEST_BASELINE_LABEL,
    }
    if reveal_paths:
        identity["source_root"] = str(_SOURCE_ROOT)
        identity["dozen_module_path"] = str(_PACKAGE_DIR)
    return identity


def startup_banner() -> str:
    """A single multi-line banner for server-side startup logs (paths shown).

    Startup logs are a trusted, local, server-side surface, so the absolute
    source root and module path are included here to make an accidental stale
    workspace obvious the moment the process boots.
    """
    identity = build_identity(reveal_paths=True)
    return (
        "[build] DOZEN runtime identity — "
        f"baseline={identity['test_baseline']} "
        f"fingerprint={identity['build_fingerprint']} "
        f"finalization_schema=v{identity['finalization_schema_version']}\n"
        f"[build] source_root={identity['source_root']}\n"
        f"[build] dozen_module={identity['dozen_module_path']}"
    )
