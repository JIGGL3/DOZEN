"""Regression coverage for the SSE run-channel lifecycle."""

from __future__ import annotations

import asyncio
import time
import unittest

from webllm import server
from webllm.progress import RunRegistry


async def _read_stream(response) -> list[str]:
    chunks: list[str] = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)
    return chunks


class TestRunStreamRetention(unittest.TestCase):
    def setUp(self) -> None:
        self._previous_runs = server.RUNS
        self.runs = RunRegistry(completed_run_retention_s=1.0)
        server.RUNS = self.runs

    def tearDown(self) -> None:
        server.RUNS = self._previous_runs
        self.runs.shutdown()

    def test_completed_stream_can_be_read_again_after_connection_closes(self) -> None:
        channel = self.runs.create()
        channel.emit({"phase": "result", "status": "done", "result": {"ok": True}})
        self.assertTrue(self.runs.close(channel.run_id))

        first = asyncio.run(_read_stream(server.run_events(channel.run_id)))
        self.assertIs(self.runs.get(channel.run_id), channel)

        second = asyncio.run(_read_stream(server.run_events(channel.run_id)))
        self.assertEqual(second, first)

    def test_completed_channel_is_removed_after_retention_window(self) -> None:
        runs = RunRegistry(completed_run_retention_s=0.01)
        try:
            channel = runs.create()
            self.assertTrue(runs.close(channel.run_id))

            deadline = time.monotonic() + 0.5
            while runs.get(channel.run_id) is not None and time.monotonic() < deadline:
                time.sleep(0.01)

            self.assertIsNone(runs.get(channel.run_id))
        finally:
            runs.shutdown()


if __name__ == "__main__":
    unittest.main()
