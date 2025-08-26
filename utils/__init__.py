"""
Utility package for helper modules used across the project.

Notes
- Having an __init__ makes imports like `import utils.opus` reliable across
  different execution contexts (scripts, modules, tests) beyond PEP 420
  namespace assumptions.
"""

__all__ = [
    # Submodules
    "opus",
    "waveform",
]
