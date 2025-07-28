import re
from pathlib import Path

# Define replacements for deprecated NumPy aliases
REPLACEMENTS = [
    (r"np\\.bool", "bool"),
    (r"np\\.object", "object"),
    (r"np\\.(int|float)", r"\1"),
    (r"dtype=np\\.float", "dtype=float"),
]

def patch_file(file_path: Path):
    """Apply compatibility patches to a file"""
    content = file_path.read_text()
    
    for pattern, replacement in REPLACEMENTS:
        content = re.sub(pattern, replacement, content)
        
    # Additional matrix to ndarray conversion
    content = content.replace("np.matrix(", "np.array(")
    
    file_path.write_text(content)
    print(f"Patched {file_path}")

def main():
    gruut_dir = Path("libs/gruut")
    
    # Patch all Python files in gruut
    for py_file in gruut_dir.glob("**/*.py"):
        patch_file(py_file)
        
    # Update version in setup.cfg
    setup_cfg = gruut_dir / "setup.cfg"
    content = setup_cfg.read_text()
    content = re.sub(r"version = \\d+\\.\\d+\\.\\d+", "version = 2.2.3.post1", content)
    setup_cfg.write_text(content)
    
    print("Gruut patching complete. Version updated to 2.2.3.post1")

if __name__ == "__main__":
    main()
