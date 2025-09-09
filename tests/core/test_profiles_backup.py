import os
from pathlib import Path


from bot.memory import profiles as profiles


def test_server_profile_backup_over_readonly_bak(tmp_path, monkeypatch):
    """
    Ensure that saving a server profile succeeds even when an existing
    .json.bak file is read-only by removing it before copying.
    """
    # Point config to temporary server profile directory
    def fake_load_config():
        return {"SERVER_PROFILE_DIR": Path(tmp_path)}

    monkeypatch.setattr("bot.config.load_config", fake_load_config, raising=True)

    # Clean cache for isolation
    profiles.server_cache.clear()

    gid = "backup_ro"

    # Step 1: Create initial profile and save once to create the JSON file
    profile = profiles.get_server_profile(gid)
    assert profile is not None
    assert profiles.save_server_profile(gid) is True

    profile_path = Path(tmp_path) / f"{gid}.json"
    bak_path = profile_path.with_suffix('.json.bak')

    assert profile_path.exists()

    # Record the current JSON content (this should become the new backup)
    before_text = profile_path.read_text(encoding='utf-8')

    # Step 2: Pre-create a read-only .bak file to simulate EACCES on overwrite
    bak_path.write_text("old backup content", encoding='utf-8')
    os.chmod(bak_path, 0o400)  # read-only
    assert bak_path.exists()

    # Step 3: Mutate profile to trigger a new save (and thus backup)
    profiles.server_cache[gid]["memories"].append("new mem")

    # Step 4: Save again - should succeed and replace the read-only .bak
    assert profiles.save_server_profile(gid) is True

    # Verify the backup now contains the previous JSON content
    bak_text = bak_path.read_text(encoding='utf-8')
    assert bak_text == before_text
