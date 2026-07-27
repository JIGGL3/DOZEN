"""Context Management System — foundation + persistence (SADD-001, Phases 1.1/1.2).

This package is the bounded context defined by
``docs/context-management-architecture.md``. Import rule (enforced by review):

    domain/  -> may import: nothing outside this package's domain/, and stdlib
    ports/   -> may import: domain/
    config/  -> may import: domain/
    utils/   -> may import: stdlib only
    adapters/ -> may import: ports/, domain/, config/, utils/

The domain never touches the filesystem, browsers, providers, or the
orchestrator pipeline. Phase 1.2 adds the filesystem persistence adapters
behind the Phase 1.1 ports; later phases (ConversationManager,
ContextPipeline) build on these contracts without modifying them.
"""

from __future__ import annotations

from .domain import enums, models, results, types
from .config.models import (
    ContextConfig,
    EstimatorConfig,
    PersistenceConfig,
    PipelineConfig,
    StorageConfig,
    SummaryPolicy,
    WindowPolicy,
)
from .adapters.filesystem import (
    FileSystemPersistenceProvider,
    RepositoryFactory,
    create_persistence,
)
from .manager import (
    ConversationManager,
    ConversationService,
)
from .context_builder import BuiltContext, ContextBuilder

__all__ = [
    "enums",
    "models",
    "results",
    "types",
    "ContextConfig",
    "EstimatorConfig",
    "PersistenceConfig",
    "PipelineConfig",
    "StorageConfig",
    "SummaryPolicy",
    "WindowPolicy",
    "FileSystemPersistenceProvider",
    "RepositoryFactory",
    "create_persistence",
    "ConversationManager",
    "ConversationService",
    "BuiltContext",
    "ContextBuilder",
]
