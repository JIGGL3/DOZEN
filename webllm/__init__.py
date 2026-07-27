"""webllm — drive logged-in LLM web UIs as the model pool for ``dozen``.

A web-automation backend that swaps ``dozen.LLMClient`` for a Playwright-driven
client. The orchestrator (Planner / Router / Executor / Verifier / Synthesizer)
is untouched; it just routes through :class:`WebAutomationLLMClient`.

Quick start
-----------
    from webllm import BrowserManager, build_orchestrator

    browser = BrowserManager(headless=False)
    browser.start_login(["openai", "anthropic"])   # opens visible windows
    # ...user logs in by hand...
    browser.confirm_login()
    orch = build_orchestrator(browser, ["openai", "anthropic"])
    print(orch.run("Compare merge sort and qudistsort.").final_answer)

Or just run the web app:  python run_web.py
"""

from .browser_cancellation import (
    DEFAULT_INTERRUPTION_POLICY,
    CancellationObservation,
    InterruptionActionResult,
    InterruptionOutcome,
    InterruptionPolicy,
    InterruptionReason,
    InterruptionRequest,
    InterruptionSnapshot,
    StopActionStatus,
)
from .browser_jobs import (
    BrowserJobEvent,
    BrowserJobId,
    BrowserJobRegistry,
    BrowserJobSnapshot,
    JobState,
    TerminalCause,
)
from .browser_manager import BrowserJobHandle, BrowserManager
from .browser_queue import (
    DEFAULT_QUEUE_POLICY,
    AdmissionOutcome,
    AdmissionSnapshot,
    BrowserQueueRejectedError,
    ProviderJobQueue,
    QueueEvent,
    QueueEventKind,
    QueuePolicy,
    QueueSnapshot,
)
from .client import WebAutomationLLMClient, render_messages
from .pool import build_orchestrator, web_pool
from .providers import PROVIDERS, ProviderAdapter, available_providers, get_adapter

__all__ = [
    "BrowserManager",
    "BrowserJobHandle",
    "BrowserJobId",
    "BrowserJobSnapshot",
    "BrowserJobEvent",
    "BrowserJobRegistry",
    "JobState",
    "TerminalCause",
    "WebAutomationLLMClient",
    "render_messages",
    "build_orchestrator",
    "web_pool",
    "PROVIDERS",
    "ProviderAdapter",
    "available_providers",
    "get_adapter",
    # Phase 5B active interruption
    "InterruptionReason",
    "InterruptionOutcome",
    "StopActionStatus",
    "InterruptionPolicy",
    "DEFAULT_INTERRUPTION_POLICY",
    "InterruptionActionResult",
    "InterruptionSnapshot",
    "InterruptionRequest",
    "CancellationObservation",
    # Phase 5C bounded queues + admission backpressure
    "QueuePolicy",
    "DEFAULT_QUEUE_POLICY",
    "AdmissionOutcome",
    "AdmissionSnapshot",
    "QueueSnapshot",
    "QueueEvent",
    "QueueEventKind",
    "ProviderJobQueue",
    "BrowserQueueRejectedError",
]
