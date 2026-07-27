"""StorageSerializer — the model <-> text codec of the persistence layer.

Implements the ``ContextSerializer`` port. All schema knowledge stays in the
domain models (``to_dict``/``from_dict`` already preserve unknown fields and
map unknown enum values to UNKNOWN); this class adds the JSON transport,
error mapping, and optional structural validation on top.
"""

from __future__ import annotations

import json
from typing import TypeVar

from ...domain.enums import ErrorCode
from ...domain.results import RepositoryResult, fail, ok

M = TypeVar("M")


class StorageSerializer:
    """JSON codec for any foundation model (anything with to_dict/from_dict).

    ``strict=False`` (the default) is deliberate: strict validation on *read*
    would reject data written by newer schemas, defeating the foundation
    layer's forward-compatibility guarantees. Strict mode exists for write
    paths that want to refuse persisting structurally broken models.
    """

    def __init__(self, strict: bool = False) -> None:
        self.strict = strict

    # ------------------------------------------------------------------ #
    # ContextSerializer port
    # ------------------------------------------------------------------ #
    def serialize(self, model: object) -> str:
        to_dict = getattr(model, "to_dict", None)
        if not callable(to_dict):
            raise TypeError(f"{type(model).__name__} has no to_dict(); not a foundation model")
        return json.dumps(to_dict(), ensure_ascii=False, separators=(",", ":"))

    def deserialize(self, payload: str, model_type: type) -> RepositoryResult[object]:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            return fail(
                ErrorCode.SERIALIZATION_FAILED, "payload is not valid JSON",
                model=model_type.__name__, position=exc.pos,
            )
        if not isinstance(data, dict):
            return fail(
                ErrorCode.SERIALIZATION_FAILED, "payload is not a JSON object",
                model=model_type.__name__,
            )
        return self.from_dict(data, model_type)

    # ------------------------------------------------------------------ #
    # Dict-level helpers used by repositories (JSONL stores dicts per line)
    # ------------------------------------------------------------------ #
    def to_dict(self, model: object) -> dict[str, object]:
        to_dict = getattr(model, "to_dict", None)
        if not callable(to_dict):
            raise TypeError(f"{type(model).__name__} has no to_dict(); not a foundation model")
        return to_dict()

    def from_dict(self, data: dict[str, object], model_type: type) -> RepositoryResult[object]:
        from_dict = getattr(model_type, "from_dict", None)
        if not callable(from_dict):
            raise TypeError(f"{model_type.__name__} has no from_dict(); not a foundation model")
        try:
            model = from_dict(data)
        except (TypeError, ValueError, KeyError) as exc:
            return fail(
                ErrorCode.SERIALIZATION_FAILED, "model deserialization failed",
                model=model_type.__name__, exception=type(exc).__name__, detail=str(exc),
            )
        if self.strict:
            problems = model.validate()
            if problems:
                return fail(
                    ErrorCode.VALIDATION_FAILED, "deserialized model failed validation",
                    model=model_type.__name__, problems=problems,
                )
        return ok(model)
