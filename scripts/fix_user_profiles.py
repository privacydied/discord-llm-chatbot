import os
import glob
import json
import shutil

PROFILE_DIR = "user_profiles"

def fix_json_file(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # Try just loading the whole file first
    try:
        json.loads(content)
        return False  # Already valid
    except json.JSONDecodeError:
        pass

    # Try to split multiple JSON objects in file
    objects = []
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(content):
        try:
            obj, end = decoder.raw_decode(content, idx)
            objects.append(obj)
            idx = end
            # Skip whitespace between objects
            while idx < len(content) and content[idx] in ' \r\n\t':
                idx += 1
        except json.JSONDecodeError:
            break

    if not objects:
        print(f"Couldn't recover any JSON objects from {path}")
        return False

    # Use the last one (most recent)
    valid = objects[-1]
    shutil.copy(path, path + ".bak")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(valid, f, indent=2, ensure_ascii=False)
    print(f"Fixed {path} (found {len(objects)} JSON objects, kept last one)")
    return True

def main():
    files = glob.glob(os.path.join(PROFILE_DIR, "*.json"))
    fixed = 0
    for fpath in files:
        if fix_json_file(fpath):
            fixed += 1
    print(f"Fixed {fixed} files.")

if __name__ == "__main__":
    main()
