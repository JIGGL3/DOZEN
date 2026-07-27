"""JSONL append-log primitives: JsonLineWriter and JsonLineReader.

Design constraints (SADD §10.2):

* Appends are the hot path — one ``open('ab')`` + write + optional fsync.
* A record is exactly one line of compact JSON terminated by ``\\n``. A crash
  mid-append can only damage the *last* line; every earlier record stays valid.
* Readers stream, never load the whole file, skip malformed lines, and report
  what they skipped (corruption detection without corruption *propagation*).
* Tail reads walk backwards in blocks so "give me the last N messages" does
  not scale with conversation length.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Sequence

from ...domain.results import RepositoryResult, ok
from .errors import io_failure

_TAIL_BLOCK_BYTES = 64 * 1024


@dataclass(frozen=True)
class JsonlLineError:
    """One malformed line, kept for diagnostics — never re-raised."""

    line_number: int  # 1-based physical line in the file
    error: str
    snippet: str  # first bytes of the offending line, for logs


@dataclass
class JsonlReadReport:
    """Result of a read: parsed records + everything that had to be skipped."""

    records: list[dict] = field(default_factory=list)
    errors: list[JsonlLineError] = field(default_factory=list)
    lines_scanned: int = 0

    @property
    def corrupt(self) -> bool:
        return bool(self.errors)


def _parse_line(raw: bytes, line_number: int) -> tuple[Optional[dict], Optional[JsonlLineError]]:
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        return None, None  # blank line: harmless, not an error
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, JsonlLineError(line_number, f"invalid json: {exc.msg}", text[:80])
    if not isinstance(value, dict):
        return None, JsonlLineError(line_number, "record is not a JSON object", text[:80])
    return value, None


class JsonLineWriter:
    """Appends records to one JSONL file. Never rewrites existing bytes."""

    def __init__(self, path: Path, fsync: bool = True) -> None:
        self.path = Path(path)
        self.fsync = fsync

    def append(self, records: Sequence[dict]) -> RepositoryResult[int]:
        """Append records; returns how many were written.

        If a previous append was interrupted mid-line (file does not end with
        a newline), a newline is written first to *seal* the damaged line so
        the new records land on clean line boundaries. The sealed partial line
        is later skipped by the reader — old data is never destroyed.
        """
        if not records:
            return ok(0)
        try:
            payload = "".join(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                for record in records
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            return io_failure(exc, "record is not JSON-serializable", path=str(self.path))
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # "a+b": append semantics for writes, but readable so the sealing
            # check below can inspect the current tail byte.
            with open(self.path, "a+b") as handle:
                if self._has_unsealed_tail(handle):
                    handle.write(b"\n")
                handle.write(payload)
                handle.flush()
                if self.fsync:
                    os.fsync(handle.fileno())
            return ok(len(records))
        except OSError as exc:
            return io_failure(exc, "jsonl append failed", path=str(self.path))

    @staticmethod
    def _has_unsealed_tail(handle) -> bool:
        """True when the file is non-empty and its last byte is not a newline
        (evidence of an interrupted append)."""
        size = handle.seek(0, os.SEEK_END)
        if size == 0:
            return False
        handle.seek(size - 1)
        last = handle.read(1)
        handle.seek(0, os.SEEK_END)
        return last != b"\n"


class JsonLineReader:
    """Streams records out of one JSONL file, tolerating damage."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def exists(self) -> bool:
        return self.path.is_file()

    def scan(self) -> Iterator[tuple[Optional[dict], Optional[JsonlLineError]]]:
        """Lazily yield ``(record, None)`` or ``(None, error)`` per line.

        A missing file yields nothing: the log of a conversation that has
        never been written to is simply empty. Raises ``OSError`` only for
        real I/O faults — callers at the repository boundary convert those.
        """
        if not self.exists():
            return
        with open(self.path, "rb") as handle:
            for line_number, raw in enumerate(handle, start=1):
                record, error = _parse_line(raw, line_number)
                if record is None and error is None:
                    continue
                yield record, error

    def read(self, offset: int = 0, limit: Optional[int] = None) -> RepositoryResult[JsonlReadReport]:
        """Read valid records with pagination (offset/limit count *valid*
        records, so pagination is stable even around corrupt lines)."""
        report = JsonlReadReport()
        skipped = 0
        try:
            for record, error in self.scan():
                report.lines_scanned += 1
                if error is not None:
                    report.errors.append(error)
                    continue
                if skipped < offset:
                    skipped += 1
                    continue
                if limit is not None and len(report.records) >= limit:
                    break
                report.records.append(record)  # type: ignore[arg-type]
            return ok(report)
        except OSError as exc:
            return io_failure(exc, "jsonl read failed", path=str(self.path))

    def count(self) -> RepositoryResult[int]:
        """Number of valid records (corrupt lines excluded)."""
        try:
            total = 0
            for record, _error in self.scan():
                if record is not None:
                    total += 1
            return ok(total)
        except OSError as exc:
            return io_failure(exc, "jsonl count failed", path=str(self.path))

    def read_tail(self, count: int) -> RepositoryResult[JsonlReadReport]:
        """Last ``count`` valid records, reading backwards in blocks — cost is
        proportional to the tail, not the file."""
        report = JsonlReadReport()
        if count <= 0:
            return ok(report)
        if not self.exists():
            return ok(report)
        try:
            with open(self.path, "rb") as handle:
                handle.seek(0, os.SEEK_END)
                position = handle.tell()
                buffer = b""
                records_rev: list[dict] = []
                while position > 0 and len(records_rev) < count:
                    step = min(_TAIL_BLOCK_BYTES, position)
                    position -= step
                    handle.seek(position)
                    buffer = handle.read(step) + buffer
                    lines = buffer.split(b"\n")
                    # lines[0] may be a partial line continuing further back;
                    # keep it in the buffer until we reach the file start.
                    keep_from = 1 if position > 0 else 0
                    buffer = lines[0] if position > 0 else b""
                    for raw in reversed(lines[keep_from:]):
                        if len(records_rev) >= count:
                            break
                        record, error = _parse_line(raw, -1)
                        if error is not None:
                            report.errors.append(error)
                        elif record is not None:
                            records_rev.append(record)
            report.records = list(reversed(records_rev[:count]))
            return ok(report)
        except OSError as exc:
            return io_failure(exc, "jsonl tail read failed", path=str(self.path))
