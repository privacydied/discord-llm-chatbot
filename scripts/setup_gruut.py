import subprocess
from pathlib import Path

GRUUT_DIR = Path("libs/gruut")
GRUUT_REPO = "https://github.com/rhasspy/gruut"

def main():
    # Ensure libs directory exists
    libs_dir = Path("libs")
    libs_dir.mkdir(exist_ok=True)
    
    # Clone gruut if not exists
    if not GRUUT_DIR.exists():
        subprocess.run(["git", "clone", GRUUT_REPO, str(GRUUT_DIR)], check=True)
    
    # Checkout tag v2.2.3
    subprocess.run(["git", "-C", str(GRUUT_DIR), "checkout", "tags/v2.2.3"], check=True)
    
    # Now run the patch_gruut.py script
    patch_script = Path(__file__).parent / "patch_gruut.py"
    subprocess.run(["python", str(patch_script)], check=True)

if __name__ == "__main__":
    main()
