"""ReliabilityRegistry — static extension-point registration (Phase 2.1.1).

This is the "new detectors / strategies / actions / failure kinds are registry
entries, not architecture changes" promise of SADD-002 §2-G7, as a concrete
object. Registration ONLY: nothing here executes, probes, retries, or touches
a browser. Later phases *read* these tables; this phase only fills them.

Not to be confused with the ``ProviderRegistry`` port (interfaces.py), which
is the provider *directory*; this registry holds the layer's plug-ins.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional

from .types import FailureType, RecoveryAction


@dataclass(frozen=True)
class FailureDescriptor:
    """Metadata for one failure kind — including future kinds that are not
    (yet) members of the FailureType enum. ``failure_type`` is UNKNOWN for
    custom registrations until the enum catches up."""

    name: str
    failure_type: FailureType = FailureType.UNKNOWN
    description: str = ""
    default_actions: tuple[RecoveryAction, ...] = ()
    terminal_for_provider: bool = False
    extra: dict[str, object] = field(default_factory=dict, repr=False)


class DuplicateRegistrationError(ValueError):
    """Raised when a name is registered twice without ``replace=True``."""


class ReliabilityRegistry:
    """Thread-safe, write-once-by-default name -> entry tables."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._detectors: dict[str, object] = {}
        self._strategies: dict[str, object] = {}
        self._actions: dict[str, RecoveryAction] = {}
        self._failures: dict[str, FailureDescriptor] = {}

    # ------------------------------ writes ----------------------------- #
    def register_detector(self, name: str, detector: object, replace: bool = False) -> None:
        self._register(self._detectors, "detector", name, detector, replace)

    def register_strategy(self, name: str, strategy: object, replace: bool = False) -> None:
        self._register(self._strategies, "strategy", name, strategy, replace)

    def register_recovery_action(
        self, name: str, action: RecoveryAction, replace: bool = False
    ) -> None:
        if not isinstance(action, RecoveryAction):
            raise TypeError(f"action must be a RecoveryAction, got {type(action).__name__}")
        self._register(self._actions, "recovery action", name, action, replace)

    def register_failure(
        self, descriptor: FailureDescriptor, replace: bool = False
    ) -> None:
        self._register(self._failures, "failure", descriptor.name, descriptor, replace)

    # ------------------------------ reads ------------------------------ #
    def detector(self, name: str) -> Optional[object]:
        with self._lock:
            return self._detectors.get(name)

    def strategy(self, name: str) -> Optional[object]:
        with self._lock:
            return self._strategies.get(name)

    def recovery_action(self, name: str) -> Optional[RecoveryAction]:
        with self._lock:
            return self._actions.get(name)

    def failure(self, name: str) -> Optional[FailureDescriptor]:
        with self._lock:
            return self._failures.get(name)

    def detectors(self) -> dict[str, object]:
        with self._lock:
            return dict(self._detectors)

    def strategies(self) -> dict[str, object]:
        with self._lock:
            return dict(self._strategies)

    def recovery_actions(self) -> dict[str, RecoveryAction]:
        with self._lock:
            return dict(self._actions)

    def failures(self) -> dict[str, FailureDescriptor]:
        with self._lock:
            return dict(self._failures)

    def snapshot(self) -> dict[str, list[str]]:
        """Registered names per table — introspection/diagnostics only."""
        with self._lock:
            return {
                "detectors": sorted(self._detectors),
                "strategies": sorted(self._strategies),
                "recovery_actions": sorted(self._actions),
                "failures": sorted(self._failures),
            }

    # ---------------------------- internals ---------------------------- #
    def _register(
        self, table: dict, kind: str, name: str, entry: object, replace: bool
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{kind} name must be a non-empty string")
        with self._lock:
            if name in table and not replace:
                raise DuplicateRegistrationError(
                    f"{kind} {name!r} is already registered (pass replace=True to override)"
                )
            table[name] = entry


# Process-wide default registry. Later phases populate it at import time with
# the built-in detectors/strategies; Phase 2.1.1 leaves it empty on purpose.
_default = ReliabilityRegistry()


def default_registry() -> ReliabilityRegistry:
    return _default
