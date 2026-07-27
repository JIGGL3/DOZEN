"""OS exception -> structured Failure mapping for the filesystem backend.

Every repository entry point catches ``OSError``/decoding errors at the
boundary and converts them into the domain's ``Failure`` values so callers
never see raw exceptions for environmental problems (missing files, permission
denied, disk full, interrupted writes). Programmer errors still raise.
"""

from __future__ import annotations

import errno

from ...domain.enums import ErrorCode
from ...domain.results import Failure, fail

# errno values that mean "the disk itself is the problem".
_DISK_FULL_ERRNOS = frozenset({errno.ENOSPC, getattr(errno, "EDQUOT", errno.ENOSPC)})


def classify_os_error(exc: BaseException) -> tuple[ErrorCode, str]:
    """Return the (code, reason) pair for an OS-level exception."""
    if isinstance(exc, FileNotFoundError):
        return ErrorCode.NOT_FOUND, "missing_file"
    if isinstance(exc, FileExistsError):
        return ErrorCode.ALREADY_EXISTS, "file_exists"
    if isinstance(exc, PermissionError):
        return ErrorCode.IO_ERROR, "permission_denied"
    if isinstance(exc, IsADirectoryError) or isinstance(exc, NotADirectoryError):
        return ErrorCode.IO_ERROR, "path_kind_mismatch"
    if isinstance(exc, OSError) and getattr(exc, "errno", None) in _DISK_FULL_ERRNOS:
        return ErrorCode.IO_ERROR, "disk_full"
    if isinstance(exc, (UnicodeDecodeError, UnicodeEncodeError)):
        return ErrorCode.SERIALIZATION_FAILED, "encoding"
    return ErrorCode.IO_ERROR, "os_error"


def io_failure(exc: BaseException, message: str, **details: object) -> Failure:
    """Build a Failure from an OS-level exception, preserving diagnostics."""
    code, reason = classify_os_error(exc)
    details.setdefault("reason", reason)
    details.setdefault("exception", type(exc).__name__)
    details.setdefault("os_message", str(exc))
    return fail(code, message, **details)
