import asyncio
import json
import tempfile
from pathlib import Path

from bot.atomic_write import atomic_write_json, read_json

def test_atomic_write_json():
    """Test basic atomic write functionality."""
    tmpdir = tempfile.mkdtemp()
    test_file = Path(tmpdir) / "test.json"
    
    data = {"key": "value", "number": 42}
    asyncio.run(atomic_write_json(test_file, data))
    
    assert test_file.exists()
    with open(test_file, "r") as f:
        loaded = json.load(f)
        assert loaded == data
    
    print("✅ atomic_write_json writes data correctly")

def test_atomic_write_preserves_json_format():
    """Test that valid JSON remains valid after atomic write."""
    tmpdir = tempfile.mkdtemp()
    test_file = Path(tmpdir) / "test.json"
    data = {"name": "Test", "value": [1, 2, 3], "nested": {"a": "b"}}
    
    asyncio.run(atomic_write_json(test_file, data))
    
    with open(test_file, "r") as f:
        loaded = json.load(f)
        assert loaded == data
    
    print("✅ atomic_write_json preserves JSON format")

def test_atomic_write_concurrent_writes():
    """Test that concurrent writes don't corrupt the file."""
    from concurrent.futures import ThreadPoolExecutor
    
    tmpdir = tempfile.mkdtemp()
    test_file = Path(tmpdir) / "test.json"
    data1 = {"value": 1}
    data2 = {"value": 2}
    data3 = {"value": 3}
    
    def write_data(data, index):
        asyncio.run(atomic_write_json(test_file, data))
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(write_data, data1, 1),
            executor.submit(write_data, data2, 2),
            executor.submit(write_data, data3, 3),
        ]
        for future in futures:
            future.result()
    
    # Only one should succeed, but the file should be valid
    with open(test_file, "r") as f:
        loaded = json.load(f)
        # Any of the three could be the final value, all are valid
        assert loaded in [data1, data2, data3]
    
    print("✅ atomic_write_json handles concurrent writes safely")

def test_read_json_tolerates_corruption():
    """Test that read_json tolerates corrupt JSON files gracefully."""
    
    tmpdir = tempfile.mkdtemp()
    test_file = Path(tmpdir) / "corrupt.json"
    # Write some corrupt JSON
    with open(test_file, "w") as f:
        f.write('{ "key": "value')  # Unterminated JSON - should be handled gracefully
    
    result = read_json(test_file, default={"fallback": True})
    assert result == {"fallback": True}
    
    print("✅ read_json tolerates corrupt JSON gracefully")

if __name__ == "__main__":
    test_atomic_write_json()
    test_atomic_write_preserves_json_format()
    test_atomic_write_concurrent_writes()
    test_read_json_tolerates_corruption()
    print("\nAll atomic write tests passed!")
