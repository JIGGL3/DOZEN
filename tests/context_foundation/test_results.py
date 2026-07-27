"""Result-type semantics: explicit success/failure, no exceptions for flow."""

from __future__ import annotations

import unittest

from dozen.context.domain.enums import ErrorCode
from dozen.context.domain.results import ErrorResult, Failure, Success, done, fail, ok


class TestResults(unittest.TestCase):
    def test_success(self) -> None:
        r = ok(42)
        self.assertTrue(r.ok)
        self.assertEqual(r.unwrap(), 42)
        self.assertEqual(r.unwrap_or(0), 42)

    def test_failure(self) -> None:
        r = fail(ErrorCode.NOT_FOUND, "no such conversation", conversation_id="c1")
        self.assertFalse(r.ok)
        self.assertEqual(r.error.code, ErrorCode.NOT_FOUND)
        self.assertEqual(r.error.details["conversation_id"], "c1")
        self.assertEqual(r.unwrap_or("fallback"), "fallback")

    def test_unwrap_failure_is_programmer_error(self) -> None:
        with self.assertRaises(ValueError):
            fail(ErrorCode.CONFLICT, "version mismatch").unwrap()

    def test_done_is_operation_success(self) -> None:
        r = done()
        self.assertTrue(r.ok)
        self.assertIsNone(r.unwrap())

    def test_error_result_serialization(self) -> None:
        err = ErrorResult(ErrorCode.IO_ERROR, "disk gone", {"path": "x"})
        again = ErrorResult.from_dict(err.to_dict())
        self.assertEqual(again, err)

    def test_error_result_unknown_code_tolerated(self) -> None:
        parsed = ErrorResult.from_dict({"code": "quantum_flux", "message": "m"})
        self.assertEqual(parsed.code, ErrorCode.UNKNOWN)

    def test_results_are_immutable(self) -> None:
        r = ok(1)
        with self.assertRaises(Exception):
            r.value = 2  # type: ignore[misc]
        f = Failure(ErrorResult(ErrorCode.UNKNOWN, "x"))
        with self.assertRaises(Exception):
            f.error = ErrorResult(ErrorCode.CONFLICT, "y")  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
