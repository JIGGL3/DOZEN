"""AtomicFileWriter — crash-safe whole-file replacement.

Protocol: write to a sibling temp file -> flush -> fsync -> ``os.replace``.
``os.replace`` is atomic on both POSIX and Windows, and the temp file lives in
the *same directory* as the target so the rename never crosses filesystems.
If anything fails before the rename, the temp file is removed and the original
file is untouched — a reader can never observe a half-written file.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from ...domain.enums import ErrorCode
from ...domain.results import OperationResult, done, fail
from .errors import io_failure


class AtomicFileWriter:
    """Replace file contents atomically; the original survives any failure."""

    def __init__(self, fsync: bool = True) -> None:
        self.fsync = fsync

    def write_text(self, path: Path, text: str) -> OperationResult:
        try:
            data = text.encode("utf-8")
        except UnicodeEncodeError as exc:
            # Lone surrogates etc.: report as a failure, never as an exception
            # (upper layers sanitize; this is the contract-keeping backstop).
            return io_failure(exc, "text is not UTF-8 encodable", path=str(path))
        return self.write_bytes(path, data)

    def write_bytes(self, path: Path, data: bytes) -> OperationResult:
        path = Path(path)
        tmp_path: str | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except FileExistsError as exc:
            # A parent path component exists but is a file — an I/O layout
            # fault, not an "already exists" business condition (which is why
            # this bypasses io_failure's FileExistsError classification).
            return fail(
                ErrorCode.IO_ERROR, "parent path is not a directory",
                path=str(path), reason="path_kind_mismatch",
                exception=type(exc).__name__, os_message=str(exc),
            )
        except OSError as exc:
            return io_failure(exc, "cannot create parent directory", path=str(path))
        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=str(path.parent), prefix=path.name + ".", suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(data)
                    handle.flush()
                    if self.fsync:
                        os.fsync(handle.fileno())
            except BaseException:
                # fd is closed by fdopen's context manager even on error.
                raise
            os.replace(tmp_path, path)
            tmp_path = None  # renamed away; nothing left to clean up
            self._fsync_directory(path.parent)
            return done()
        except (OSError, ValueError) as exc:
            return io_failure(exc, "atomic write failed", path=str(path))
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass  # best-effort cleanup; stray *.tmp files are inert

    def _fsync_directory(self, directory: Path) -> None:
        """Persist the rename itself (POSIX). Windows cannot open directories
        with os.open, so this is best-effort by design."""
        if not self.fsync:
            return
        try:
            dir_fd = os.open(str(directory), os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(dir_fd)
        except OSError:
            pass
        finally:
            os.close(dir_fd)
