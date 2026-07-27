"""Result types — explicit success/failure instead of exceptions for control flow.

Repository and pipeline operations return ``Result`` values; exceptions are
reserved for programmer errors (e.g. unwrapping a Failure) and cancellation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar, Union

from .enums import ErrorCode

T = TypeVar("T")


@dataclass(frozen=True)
class ErrorResult:
    """A structured, serializable error."""

    code: ErrorCode
    message: str
    details: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code.value,
            "message": self.message,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ErrorResult":
        raw_code = str(data.get("code", ErrorCode.UNKNOWN.value))
        try:
            code = ErrorCode(raw_code)
        except ValueError:
            code = ErrorCode.UNKNOWN
        details = data.get("details")
        return cls(
            code=code,
            message=str(data.get("message", "")),
            details=dict(details) if isinstance(details, dict) else {},
        )


@dataclass(frozen=True)
class Success(Generic[T]):
    value: T

    @property
    def ok(self) -> bool:
        return True

    def unwrap(self) -> T:
        return self.value

    def unwrap_or(self, default: T) -> T:
        return self.value


@dataclass(frozen=True)
class Failure:
    error: ErrorResult

    @property
    def ok(self) -> bool:
        return False

    def unwrap(self) -> object:
        # Unwrapping a Failure is a programmer error, not normal control flow.
        raise ValueError(
            f"Called unwrap() on Failure({self.error.code.value}: {self.error.message})"
        )

    def unwrap_or(self, default: T) -> T:
        return default


# A Result is either Success[T] or Failure. Aliases document intent at call sites.
Result = Union[Success[T], Failure]
RepositoryResult = Union[Success[T], Failure]
OperationResult = Union[Success[None], Failure]


def ok(value: T) -> Success[T]:
    return Success(value)


def done() -> Success[None]:
    """An OperationResult success with no payload."""
    return Success(None)


def fail(code: ErrorCode, message: str, **details: object) -> Failure:
    return Failure(ErrorResult(code=code, message=message, details=dict(details)))
