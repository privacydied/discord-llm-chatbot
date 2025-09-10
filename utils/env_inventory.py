#!/usr/bin/env python3
"""
Environment Variable Inventory Utility

Scans the repository for environment variable usage and outputs a JSON report.
- Detects: os.getenv, os.environ.get, os.environ["KEY"]
- Captures: key, default, usage type, file, line, coercion hints (int/float/bool), notes
- Logging: Pretty (RichHandler) + JSONL sink with enforcer

Usage:
  uv run python utils/env_inventory.py --root . --out utils/env_inventory.json

[PA][REH][IV][CMV][CA][CSD]
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from rich.logging import RichHandler
    from rich.console import Console
    from rich.panel import Panel
    from rich.tree import Tree
except Exception:  # pragma: no cover
    print(
        "Rich is required for this utility. Please install `rich`.\n", file=sys.stderr
    )
    raise


# -----------------------------
# Logging setup with dual sinks
# -----------------------------
class JSONLFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ts = (
            datetime.fromtimestamp(record.created)
            .astimezone()
            .strftime("%Y-%m-%dT%H:%M:%S.%f%z")
        )
        payload = {
            "ts": ts[:-8] + ts[-5:],  # ms precision alignment
            "level": record.levelname,
            "name": record.name,
            "subsys": "env_inventory",
            "guild_id": None,
            "user_id": None,
            "msg_id": None,
            "event": getattr(record, "event", None),
            "detail": record.getMessage(),
        }
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)

    pretty_handler = RichHandler(
        rich_tracebacks=False,
        markup=True,
        show_time=True,
        show_path=False,
        log_time_format="%H:%M:%S.%f",
        omit_repeated_times=False,
    )
    pretty_handler.setLevel(logging.INFO)

    jsonl_handler = logging.FileHandler(
        log_dir / "env_inventory.jsonl", encoding="utf-8"
    )
    jsonl_handler.setLevel(logging.DEBUG)
    jsonl_handler.setFormatter(JSONLFormatter())

    logger = logging.getLogger("utils.env_inventory")
    logger.setLevel(logging.DEBUG)
    logger.handlers = [pretty_handler, jsonl_handler]
    logger.propagate = False

    # Enforcer: two handlers must be active [SFT]
    handlers = logger.handlers
    if not any(isinstance(h, RichHandler) for h in handlers) or not any(
        isinstance(h, logging.FileHandler) for h in handlers
    ):
        print(
            "✖ Logging enforcer failed: missing Pretty or JSONL handler",
            file=sys.stderr,
        )
        sys.exit(2)

    logger.debug("Logging configured with Pretty and JSONL sinks")
    return logger


# -----------------------------
# AST scanning
# -----------------------------
@dataclass
class EnvUse:
    key: str
    source: str  # getenv | environ_get | environ_index
    file: str
    line: int
    default: Optional[str] = None
    coerced_as: Optional[str] = None  # int | float | bool | list | set | none
    notes: Optional[str] = None


class ParentTrackingNodeVisitor(ast.NodeVisitor):
    """AST visitor that tracks parent stack for context analysis."""

    def __init__(self):
        super().__init__()
        self.parent_stack: List[ast.AST] = []

    def visit(self, node: ast.AST):
        self.parent_stack.append(node)
        try:
            return super().visit(node)
        finally:
            self.parent_stack.pop()

    def get_parent(self, n_back: int = 1) -> Optional[ast.AST]:
        if len(self.parent_stack) > n_back:
            return self.parent_stack[-(n_back + 1)]
        return None


class EnvScanner(ParentTrackingNodeVisitor):
    def __init__(self, file_path: Path, logger: logging.Logger):
        super().__init__()
        self.file_path = file_path
        self.logger = logger
        self.uses: List[EnvUse] = []

    def _coercion_hint(self) -> Optional[str]:
        # Look at immediate parent call like int(...), float(...), bool(...)
        parent = self.get_parent(1)
        if isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name):
            name = parent.func.id
            if name in {"int", "float", "bool", "list", "set"}:
                return name
        return None

    def _extract_str(self, node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return None

    def visit_Call(self, node: ast.Call):
        # os.getenv("KEY", default)
        if isinstance(node.func, ast.Attribute) and isinstance(
            node.func.value, ast.Name
        ):
            if node.func.value.id == "os" and node.func.attr == "getenv":
                if node.args:
                    key = self._extract_str(node.args[0])
                    default = None
                    if len(node.args) > 1:
                        default = self._literal_repr(node.args[1])
                    for kw in node.keywords or []:
                        if kw.arg == "default":
                            default = self._literal_repr(kw.value)
                    if key:
                        self._record(key, "getenv", node, default)
        # os.environ.get("KEY", default)
        if isinstance(node.func, ast.Attribute) and isinstance(
            node.func.value, ast.Subscript
        ):
            pass  # not our case
        if isinstance(node.func, ast.Attribute) and isinstance(
            node.func.value, ast.Attribute
        ):
            # os.environ.get
            if (
                isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id == "os"
                and node.func.value.attr == "environ"
                and node.func.attr == "get"
            ):
                if node.args:
                    key = self._extract_str(node.args[0])
                    default = None
                    if len(node.args) > 1:
                        default = self._literal_repr(node.args[1])
                    if key:
                        self._record(key, "environ_get", node, default)
        return self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript):
        # os.environ["KEY"]
        if isinstance(node.value, ast.Attribute) and isinstance(
            node.value.value, ast.Name
        ):
            if node.value.value.id == "os" and node.value.attr == "environ":
                key = None
                if isinstance(node.slice, ast.Constant) and isinstance(
                    node.slice.value, str
                ):
                    key = node.slice.value
                elif isinstance(node.slice, ast.Index) and isinstance(
                    node.slice.value, ast.Constant
                ):  # py<3.9
                    if isinstance(node.slice.value.value, str):
                        key = node.slice.value.value
                if key:
                    self._record(key, "environ_index", node, None)
        return self.generic_visit(node)

    def _literal_repr(self, node: ast.AST) -> Optional[str]:
        try:
            if isinstance(node, ast.Constant):
                return repr(node.value)
            if isinstance(node, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
                return ast.unparse(node)  # py3.9+
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in {"int", "float", "bool", "str"}
            ):
                # show callable default like int("5")
                return ast.unparse(node)
            # fallback generic
            return ast.unparse(node)
        except Exception:
            return None

    def _record(self, key: str, source: str, node: ast.AST, default: Optional[str]):
        coerced = self._coercion_hint()
        use = EnvUse(
            key=key,
            source=source,
            file=str(self.file_path.relative_to(Path.cwd())),
            line=getattr(node, "lineno", -1),
            default=default,
            coerced_as=coerced,
            notes=None,
        )
        self.uses.append(use)


# -----------------------------
# Core logic
# -----------------------------
SENSITIVE_HINTS = (
    "TOKEN",
    "SECRET",
    "KEY",
    "PASSWORD",
    "BEARER",
    "API_KEY",
    "AUTH",
    "WEBHOOK",
)


def classify_sensitivity(key: str) -> str:
    u = key.upper()
    return "sensitive" if any(h in u for h in SENSITIVE_HINTS) else "normal"


def scan_repo(root: Path, logger: logging.Logger) -> Dict[str, List[Dict[str, Any]]]:
    results: Dict[str, List[Dict[str, Any]]] = {}
    for dirpath, _, filenames in os.walk(root):
        if any(
            part in {".git", ".venv", "node_modules", "logs"}
            for part in Path(dirpath).parts
        ):
            continue
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            p = Path(dirpath) / fn
            try:
                src = p.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning(f"Skipping unreadable file {p}: {e}")
                continue
            try:
                tree = ast.parse(src)
            except SyntaxError as e:
                logger.warning(f"Skipping file with syntax error {p}: {e}")
                continue

            scanner = EnvScanner(p, logger)
            scanner.visit(tree)

            for use in scanner.uses:
                results.setdefault(use.key, []).append(asdict(use))
    return results


def summarize(results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for key, uses in results.items():
        defaults = sorted(
            {u.get("default") for u in uses if u.get("default") is not None}
        )
        coercions = sorted({u.get("coerced_as") for u in uses if u.get("coerced_as")})
        summary[key] = {
            "uses": uses,
            "count": len(uses),
            "defaults": defaults,
            "coercions": coercions,
            "sensitivity": classify_sensitivity(key),
        }
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inventory environment variable usage across the repo"
    )
    parser.add_argument(
        "--root", type=str, default=str(Path.cwd()), help="Repository root to scan"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="utils/env_inventory.json",
        help="Output JSON file path",
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    out_path = Path(args.out)

    logger = setup_logging(Path("logs"))

    console = Console()
    console.print(
        Panel.fit(
            f"Starting env inventory scan in [bold]{root}[/bold]", title="Env Inventory"
        )
    )

    if not root.exists():
        logger.error(f"Root path does not exist: {root}")
        return 2

    results = scan_repo(root, logger)
    report = summarize(results)

    # Ensure output directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Pretty report tree
    tree = Tree("Environment Variables")
    for key in sorted(report.keys()):
        node = tree.add(
            f"[bold]{key}[/bold] x{report[key]['count']} [{report[key]['sensitivity']}]"
        )
        defs = (
            ", ".join(d for d in report[key]["defaults"] if d is not None) or "(none)"
        )
        coers = ", ".join(report[key]["coercions"]) or "(none)"
        node.add(f"defaults: {defs}")
        node.add(f"coercions: {coers}")
    console.print(tree)

    logger.info(f"✔ Wrote env inventory to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
