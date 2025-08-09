#!/usr/bin/env python3
"""
Environment Drift Guard

Compares three sources of truth and fails with non-zero exit if drift is detected:
- utils/env_inventory.json (AST scan of code usage)
- docs/config/ENV_REGISTRY.json (canonical registry generated)
- configs/.env.sample (distributed sample env file)

Checks:
1) Missing in registry: keys used in code but absent from registry [FAIL]
2) Missing in sample: keys in registry but absent from .env.sample [FAIL]
3) Unused in code: keys in registry but not referenced in code [configurable FAIL]
4) Sensitive defaults in sample: sensitive keys must have blank values [FAIL]

Flags:
--ignore-keys: comma-separated list to ignore from all checks
--fail-on-unused: default true; if false, unused keys only WARN

Outputs report to logs/env_drift_report.json.

[IV][CMV][SFT][PA][REH][CA]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from rich.logging import RichHandler
from rich.panel import Panel
from rich.console import Console


# -----------------------------
# Logging (Dual Sink + Enforcer)
# -----------------------------
class JSONLFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created).astimezone().strftime("%Y-%m-%dT%H:%M:%S.%f%z")
        payload = {
            "ts": ts[:-8] + ts[-5:],
            "level": record.levelname,
            "name": record.name,
            "subsys": "env_drift_guard",
            "guild_id": None,
            "user_id": None,
            "msg_id": None,
            "event": getattr(record, "event", None),
            "detail": record.getMessage(),
        }
        return json.dumps(payload, ensure_ascii=False)


def get_logger() -> logging.Logger:
    pretty_handler = RichHandler(
        rich_tracebacks=False,
        markup=True,
        show_time=True,
        show_path=False,
        log_time_format="%H:%M:%S.%f",
        omit_repeated_times=False,
    )
    pretty_handler.setLevel(logging.INFO)

    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    jsonl_handler = logging.FileHandler(logs_dir / "env_drift_guard.jsonl", encoding="utf-8")
    jsonl_handler.setLevel(logging.DEBUG)
    jsonl_handler.setFormatter(JSONLFormatter())

    logger = logging.getLogger("utils.env_drift_guard")
    logger.setLevel(logging.DEBUG)
    logger.handlers = [pretty_handler, jsonl_handler]
    logger.propagate = False

    # Enforcer: ensure both pretty and json handlers
    if not any(isinstance(h, RichHandler) for h in logger.handlers) or not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        print("✖ Logging enforcer failed: missing Pretty or JSONL handler")
        raise SystemExit(2)

    return logger


# -----------------------------
# Data Models
# -----------------------------
@dataclass
class RegistryItem:
    key: str
    group: str
    description: str
    required: bool
    sensitive: bool
    type: str
    default: Optional[str]
    all_defaults: List[str]
    valid_values: Optional[str]


# -----------------------------
# Utilities
# -----------------------------
KEY_RE = re.compile(r"^([A-Z0-9_]+)=(.*)$")


def parse_env_sample(path: Path) -> Dict[str, str]:
    result: Dict[str, str] = {}
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = KEY_RE.match(line)
        if m:
            key, value = m.group(1), m.group(2)
            result[key] = value
    return result


def load_registry(path: Path) -> Dict[str, RegistryItem]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items: Dict[str, RegistryItem] = {}
    for grp, lst in data.items():
        for it in lst:
            item = RegistryItem(**it)
            items[item.key] = item
    return items


def load_inventory(path: Path) -> Set[str]:
    inv = json.loads(path.read_text(encoding="utf-8"))
    return set(inv.keys())


# -----------------------------
# Drift Checks
# -----------------------------
@dataclass
class DriftReport:
    missing_in_registry: List[str]
    missing_in_sample: List[str]
    unused_in_code: List[str]
    sensitive_nonblank_in_sample: List[str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


def compute_drift(
    inv_keys: Set[str],
    reg_items: Dict[str, RegistryItem],
    sample_map: Dict[str, str],
    ignore: Set[str],
    fail_on_unused: bool,
) -> Tuple[DriftReport, bool]:
    # Apply ignore
    inv_keys = {k for k in inv_keys if k not in ignore}
    reg_keys = {k for k in reg_items.keys() if k not in ignore}
    sample_keys = {k for k in sample_map.keys() if k not in ignore}

    missing_in_registry = sorted(inv_keys - reg_keys)
    missing_in_sample = sorted(reg_keys - sample_keys)
    unused_in_code = sorted(reg_keys - inv_keys)

    sensitive_nonblank_in_sample: List[str] = []
    for k in sorted(reg_keys & sample_keys):
        item = reg_items.get(k)
        if item and item.sensitive:
            val = (sample_map.get(k) or "").strip()
            if val:
                sensitive_nonblank_in_sample.append(k)

    # Determine failure
    fail = False
    if missing_in_registry:
        fail = True
    if missing_in_sample:
        fail = True
    if sensitive_nonblank_in_sample:
        fail = True
    if fail_on_unused and unused_in_code:
        fail = True

    report = DriftReport(
        missing_in_registry=missing_in_registry,
        missing_in_sample=missing_in_sample,
        unused_in_code=unused_in_code,
        sensitive_nonblank_in_sample=sensitive_nonblank_in_sample,
    )
    return report, fail


# -----------------------------
# Main
# -----------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Environment config drift guard")
    parser.add_argument("--inventory", default="utils/env_inventory.json")
    parser.add_argument("--registry-json", default="docs/config/ENV_REGISTRY.json")
    parser.add_argument("--env-sample", default="configs/.env.sample")
    parser.add_argument("--ignore-keys", default="", help="Comma-separated list of keys to ignore")
    parser.add_argument("--fail-on-unused", action="store_true", default=True)
    parser.add_argument("--allow-unused", dest="fail_on_unused", action="store_false")
    args = parser.parse_args()

    logger = get_logger()
    console = Console()
    console.print(Panel.fit("Running Env Drift Guard", title="Env Drift"))

    inv_path = Path(args.inventory)
    reg_path = Path(args.registry_json)
    sample_path = Path(args.env_sample)

    missing = []
    for p in [inv_path, reg_path, sample_path]:
        if not p.exists():
            missing.append(str(p))
    if missing:
        logger.error(f"Missing required files: {missing}")
        return 2

    ignore = {k.strip() for k in args.ignore_keys.split(",") if k.strip()}

    inv_keys = load_inventory(inv_path)
    reg_items = load_registry(reg_path)
    sample_map = parse_env_sample(sample_path)

    report, fail = compute_drift(inv_keys, reg_items, sample_map, ignore, args.fail_on_unused)

    # Persist JSON report
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "env_drift_report.json").write_text(report.to_json(), encoding="utf-8")

    # Log findings succinctly
    if report.missing_in_registry:
        logger.error(f"Keys used in code but MISSING from registry: {report.missing_in_registry}")
    if report.missing_in_sample:
        logger.error(f"Keys in registry but MISSING from .env.sample: {report.missing_in_sample}")
    if report.sensitive_nonblank_in_sample:
        logger.error(f"Sensitive keys with NON-BLANK values in sample: {report.sensitive_nonblank_in_sample}")
    if report.unused_in_code:
        level = logging.ERROR if args.fail_on_unused else logging.WARNING
        logger.log(level, f"Keys in registry but UNUSED in code: {report.unused_in_code}")

    if fail:
        logger.error("✖ Drift detected. Failing.")
        return 1

    logger.info("✔ No drift detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
