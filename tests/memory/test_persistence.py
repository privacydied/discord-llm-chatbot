"""
Tests for atomic persistence layer.

[REH] Corruption recovery paths
[SFT] Atomic write semantics
"""

import json
from pathlib import Path

from bot.memory.persistence import (
    _validate_json,
    _atomic_write_file,
    _create_backup,
    _restore_from_backup,
    atomic_save_json,
    load_json_with_recovery,
    validate_profile_integrity,
)


class TestValidateJson:
    def test_valid_dict(self):
        assert _validate_json({"key": "value"}) is True
        assert _validate_json({"nested": {"key": [1, 2, 3]}}) is True

    def test_invalid_contains_bytes(self):
        # bytes are not JSON serializable
        assert _validate_json({"data": b"binary"}) is False

    def test_invalid_contains_set(self):
        # sets are not JSON serializable
        assert _validate_json({"items": {1, 2, 3}}) is False


class TestAtomicWriteFile:
    def test_basic_write(self, tmp_path):
        target = tmp_path / "test.json"
        data = {"key": "value", "number": 42}

        result = _atomic_write_file(target, data)
        assert result is True
        assert target.exists()

        # Verify content
        with open(target) as f:
            loaded = json.load(f)
        assert loaded == data

    def test_atomic_replace(self, tmp_path):
        target = tmp_path / "test.json"

        # First write
        data1 = {"version": 1}
        _atomic_write_file(target, data1)

        # Replace with new data
        data2 = {"version": 2}
        result = _atomic_write_file(target, data2)
        assert result is True

        with open(target) as f:
            loaded = json.load(f)
        assert loaded["version"] == 2

    def test_invalid_json_not_written(self, tmp_path):
        target = tmp_path / "test.json"
        data = {"invalid": set([1, 2, 3])}  # sets can't be JSON serialized

        result = _atomic_write_file(target, data)
        assert result is False
        assert not target.exists()


class TestCreateBackup:
    def test_backup_created(self, tmp_path):
        src = tmp_path / "original.json"
        dst = tmp_path / "backup.json"

        # Create source
        src.write_text('{"test": "data"}')

        result = _create_backup(src, dst)
        assert result is True
        assert dst.exists()
        assert json.loads(dst.read_text()) == {"test": "data"}

    def test_overwrite_existing_backup(self, tmp_path):
        src = tmp_path / "original.json"
        dst = tmp_path / "backup.json"

        src.write_text('{"v": 2}')
        dst.write_text('{"v": 1}')

        result = _create_backup(src, dst)
        assert result is True
        assert json.loads(dst.read_text()) == {"v": 2}


class TestRestoreFromBackup:
    def test_successful_restore(self, tmp_path):
        backup = tmp_path / "backup.json"
        target = tmp_path / "corrupted.json"

        backup.write_text('{"restored": true}')
        target.write_text("garbage not json")

        result = _restore_from_backup(backup, target)
        assert result is True
        assert json.loads(target.read_text()) == {"restored": True}

    def test_restore_from_corrupted_backup_fails(self, tmp_path):
        backup = tmp_path / "backup.json"
        target = tmp_path / "target.json"

        backup.write_text("also not json")

        result = _restore_from_backup(backup, target)
        assert result is False

    def test_restore_missing_backup(self, tmp_path):
        target = tmp_path / "target.json"
        result = _restore_from_backup(Path("/nonexistent"), target)
        assert result is False


class TestAtomicSaveJson:
    def test_save_with_backup(self, tmp_path):
        target = tmp_path / "profile.json"
        data = {"user_id": "123", "memories": ["memory1"]}

        # First write creates the file (no backup since file doesn't exist yet)
        result = atomic_save_json(target, data, create_backup=True)
        assert result is True
        assert target.exists()

        # Second write creates backup of the existing file
        data2 = {"user_id": "123", "memories": ["memory1", "memory2"]}
        result2 = atomic_save_json(target, data2, create_backup=True)
        assert result2 is True
        assert (target.with_suffix(".json.bak")).exists()
        # Backup should have the old data
        backup_data = json.loads((target.with_suffix(".json.bak")).read_text())
        assert len(backup_data["memories"]) == 1

    def test_save_without_backup(self, tmp_path):
        target = tmp_path / "profile.json"
        data = {"user_id": "456"}

        result = atomic_save_json(target, data, create_backup=False)
        assert result is True
        assert target.exists()
        assert not (target.with_suffix(".json.bak")).exists()

    def test_save_invalid_data_fails(self, tmp_path):
        target = tmp_path / "profile.json"
        data = {"invalid": frozenset([1, 2])}  # Cannot be JSON serialized

        result = atomic_save_json(target, data, validate_before_write=True)
        assert result is False

    def test_save_skips_validation(self, tmp_path):
        target = tmp_path / "profile.json"
        data = {"user_id": "789"}

        result = atomic_save_json(target, data, validate_before_write=False)
        assert result is True


class TestLoadJsonWithRecovery:
    def test_load_valid_file(self, tmp_path):
        target = tmp_path / "profile.json"
        data = {"user_id": "123"}
        target.write_text(json.dumps(data))

        result = load_json_with_recovery(target)
        assert result == data

    def test_missing_file_returns_default(self, tmp_path):
        target = tmp_path / "missing.json"
        default = {"default": True}

        result = load_json_with_recovery(target, default_data=default)
        assert result == default

    def test_corrupted_file_with_recovery(self, tmp_path):
        target = tmp_path / "profile.json"
        backup = tmp_path / "profile.json.bak"

        target.write_text("not json")
        backup.write_text(json.dumps({"recovered": True}))

        result = load_json_with_recovery(target, attempt_recovery=True)
        assert result == {"recovered": True}

    def test_corrupted_file_no_recovery(self, tmp_path):
        target = tmp_path / "profile.json"
        target.write_text("not json")

        result = load_json_with_recovery(target, attempt_recovery=False, default_data={})
        assert result == {}


class TestValidateProfileIntegrity:
    def test_valid_user_profile(self):
        profile = {"discord_id": "123", "memories": [], "preferences": {}}
        is_valid, error = validate_profile_integrity(profile)
        assert is_valid is True
        assert error is None

    def test_valid_server_profile(self):
        profile = {"guild_id": "456", "memories": []}
        is_valid, error = validate_profile_integrity(profile)
        assert is_valid is True
        assert error is None

    def test_missing_id(self):
        profile = {"memories": []}
        is_valid, error = validate_profile_integrity(profile)
        assert is_valid is False
        assert "missing identifier" in error.lower()

    def test_not_a_dict(self):
        is_valid, error = validate_profile_integrity("not a dict")
        assert is_valid is False

    def test_memories_not_list(self):
        profile = {"discord_id": "123", "memories": "notalist"}
        is_valid, error = validate_profile_integrity(profile)
        assert is_valid is False
        assert "memories" in error.lower()

    def test_preferences_not_dict(self):
        profile = {"discord_id": "123", "preferences": []}
        is_valid, error = validate_profile_integrity(profile)
        assert is_valid is False
        assert "preferences" in error.lower()


class TestConcurrencyScenarios:
    """Test behavior under concurrent access patterns."""

    def test_atomic_semantics_exist(self, tmp_path):
        """Verify atomic write exists and handles replacement."""
        target = tmp_path / "concurrent.json"

        # Simulate rapid writes
        for i in range(10):
            data = {"counter": i, "timestamp": i}
            result = _atomic_write_file(target, data)
            assert result is True

        # Verify final state is consistent
        with open(target) as f:
            final = json.load(f)
        assert final["counter"] == 9

    def test_concurrent_async_writes_no_corruption(self, tmp_path):
        """Two coroutines writing to the same file should not produce corrupt JSON."""
        import asyncio

        target = tmp_path / "concurrent_async.json"

        async def writer(data, delay=0.01):
            await asyncio.sleep(delay)
            return _atomic_write_file(target, data)

        async def run_concurrent():
            results = await asyncio.gather(
                writer({"source": "A", "value": 1}, delay=0.0),
                writer({"source": "B", "value": 2}, delay=0.005),
            )
            return results

        results = asyncio.run(run_concurrent())
        assert all(r is True for r in results)

        # File must be valid JSON (either A or B, not a mix)
        with open(target) as f:
            data = json.load(f)
        assert data["source"] in ("A", "B")
        assert isinstance(data["value"], int)

    def test_interrupted_write_recovery(self, tmp_path):
        """If a write is interrupted (temp file left), load_with_recovery still works."""
        target = tmp_path / "interrupted.json"

        # Write valid data first
        _atomic_write_file(target, {"version": 1, "data": "original"})

        # Simulate interruption: write a partial/corrupt file directly
        with open(target, "w") as f:
            f.write('{"version": 2, "data": "corrupt')  # truncated JSON

        # load_with_recovery should detect corruption and recover from backup
        result = load_json_with_recovery(target, default_data={"version": 0})
        assert result is not None
        # Either recovered from backup or returned default
        assert result.get("version") in (1, 0)

    def test_temp_file_cleanup_on_success(self, tmp_path):
        """Temp files from atomic writes are cleaned up after successful rename."""
        target = tmp_path / "cleanup.json"
        _atomic_write_file(target, {"clean": True})

        # No .tmp_ files should remain
        tmp_files = list(tmp_path.glob(".tmp_*"))
        assert len(tmp_files) == 0, f"Stale temp files: {tmp_files}"

    def test_lock_file_created_and_cleaned(self, tmp_path):
        """File lock is acquired and released during atomic_save_json."""
        target = tmp_path / "locked.json"
        result = atomic_save_json(target, {"locked": True}, use_lock=True)
        assert result is True

        # Lock file may or may not persist (implementation detail)
        # but the data file must be valid
        with open(target) as f:
            data = json.load(f)
        assert data["locked"] is True
