"""Tests for bot/atomic_json.py — atomic JSON write helper.

Covers:
- Concurrent async writes to the same file (per-path lock)
- Interrupted/corrupt temp file recovery
- Valid JSON remains valid after concurrent writes
- read_json_safe tolerates corrupt/missing/empty files
- Parent directory creation
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from bot.atomic_json import read_json_safe, write_json_atomic


@pytest.fixture
def tmp_dir():
    """Provide a clean temporary directory per test."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# ------------------------------------------------------------------ #
# write_json_atomic — basic correctness
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_write_and_read_roundtrip(tmp_dir):
    path = tmp_dir / "test.json"
    data = {"key": "value", "number": 42}

    await write_json_atomic(path, data)
    result = read_json_safe(path)

    assert result == data


@pytest.mark.asyncio
async def test_creates_parent_directories(tmp_dir):
    path = tmp_dir / "a" / "b" / "c" / "data.json"
    data = {"nested": True}

    await write_json_atomic(path, data)
    result = read_json_safe(path)

    assert result == data


@pytest.mark.asyncio
async def test_overwrites_existing(tmp_dir):
    path = tmp_dir / "overwrite.json"

    await write_json_atomic(path, {"v": 1})
    await write_json_atomic(path, {"v": 2})

    assert read_json_safe(path) == {"v": 2}


# ------------------------------------------------------------------ #
# read_json_safe — error tolerance
# ------------------------------------------------------------------ #


def test_read_missing_file_returns_default(tmp_dir):
    path = tmp_dir / "does_not_exist.json"
    assert read_json_safe(path, default={"fallback": True}) == {"fallback": True}


def test_read_corrupt_json_returns_default(tmp_dir, caplog):
    path = tmp_dir / "corrupt.json"
    path.write_text("{not valid json!!!", encoding="utf-8")

    result = read_json_safe(path, default={"safe": True})
    assert result == {"safe": True}
    assert "Corrupt JSON" in caplog.text


def test_read_empty_file_returns_default(tmp_dir, caplog):
    path = tmp_dir / "empty.json"
    path.write_text("", encoding="utf-8")

    result = read_json_safe(path, default={"safe": True})
    assert result == {"safe": True}
    assert "Empty JSON file" in caplog.text


def test_read_whitespace_only_returns_default(tmp_dir):
    path = tmp_dir / "ws.json"
    path.write_text("  \n  \t  ", encoding="utf-8")

    assert read_json_safe(path, default="ok") == "ok"


# ------------------------------------------------------------------ #
# Concurrent writes to same file (per-path lock)
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_concurrent_writes_same_file_no_corruption(tmp_dir):
    """10 concurrent writers to the same file should never leave corrupt JSON."""
    path = tmp_dir / "concurrent.json"
    iterations = 10

    async def writer(n: int):
        await write_json_atomic(path, {"iteration": n, "items": list(range(n))})

    await asyncio.gather(*(writer(i) for i in range(iterations)))

    # File must be valid JSON after all writes complete
    result = read_json_safe(path)
    assert result is not None
    assert "iteration" in result
    assert "items" in result
    assert isinstance(result["items"], list)


@pytest.mark.asyncio
async def test_concurrent_writes_same_file_final_value_valid(tmp_dir):
    """After many concurrent writes, the final file must hold one complete write."""
    path = tmp_dir / "race.json"
    n_writers = 20

    async def writer(n: int):
        await write_json_atomic(path, {"writer": n, "value": "x" * 1000})

    await asyncio.gather(*(writer(i) for i in range(n_writers)))

    result = read_json_safe(path)
    assert result is not None
    assert result["writer"] in range(n_writers)
    assert len(result["value"]) == 1000


@pytest.mark.asyncio
async def test_concurrent_writes_different_files_no_interference(tmp_dir):
    """Writes to different paths must not corrupt each other."""
    paths = [tmp_dir / f"f_{i}.json" for i in range(5)]
    data = [{"id": i, "data": [i] * 10} for i in range(5)]

    tasks = [write_json_atomic(p, d) for p, d in zip(paths, data)]
    await asyncio.gather(*tasks)

    for p, d in zip(paths, data):
        assert read_json_safe(p) == d


# ------------------------------------------------------------------ #
# Temp file cleanup on failure
# ------------------------------------------------------------------ #


def _os_replace_failing_raise(tmp_path: str, final_path: str):
    raise OSError("simulated replace failure")


@pytest.mark.asyncio
async def test_temp_file_cleaned_up_on_write_failure(tmp_dir):
    """If os.replace fails, the temp file should be cleaned up."""
    path = tmp_dir / "fail.json"

    before_files = set(os.listdir(tmp_dir))

    with patch("bot.atomic_json.os.replace", side_effect=OSError("fail")):
        with pytest.raises(OSError, match="fail"):
            await write_json_atomic(path, {"should": "not persist"})

    after_files = set(os.listdir(tmp_dir))
    # No leftover .tmp files
    leftovers = after_files - before_files
    assert len(leftovers) == 0


# ------------------------------------------------------------------ #
# Corrupt temp file recovery — simulates interrupted write leaving
# a .tmp file that read_json_safe must ignore
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_leftover_temp_file_does_not_corrupt_read(tmp_dir):
    """A stale .tmp file from a previous interrupted write should not affect reads."""

    # Simulate a leftover temp file
    final_path = tmp_dir / "data.json"
    tmp_path = tmp_dir / ".data.json.tmp12345"
    tmp_path.write_text("{corrupt!!!", encoding="utf-8")

    # Write valid data to final path
    await write_json_atomic(final_path, {"status": "ok"})

    # read_json_safe on the final path should return valid data
    result = read_json_safe(final_path)
    assert result == {"status": "ok"}


# ------------------------------------------------------------------ #
# Large file write (ensures fsync completes)
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_large_write_remains_valid(tmp_dir):
    """A large JSON write must produce valid JSON after fsync."""
    path = tmp_dir / "large.json"
    data = {"big_list": [i for i in range(10000)], "nested": {"a": [1] * 500}}

    await write_json_atomic(path, data)
    result = read_json_safe(path)

    assert result == data
    assert len(result["big_list"]) == 10000
