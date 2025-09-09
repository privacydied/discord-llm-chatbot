#!/usr/bin/env python3
"""
Builds a canonical environment variable registry and .env.sample from utils/env_inventory.json.

Generates:
- configs/.env.sample
- docs/config/ENV_REGISTRY.md

Heuristics:
- Group by prefix before first underscore (e.g., RAG_, STT_, PROMETHEUS_, X_API_, STREAMING_, TTS_, WEBEX_, MEDIA_, VIDEO_). Ungrouped -> CORE.
- Required set (initial): {DISCORD_TOKEN, PROMPT_FILE, VL_PROMPT_FILE}
- Sensitive detection via keyword hints.
- Type from coercion hints; else unknown.
- Default: consolidate from inventory; choose first non-None as canonical default; if multiple conflicting defaults -> leave blank and list in notes.

[PA][REH][IV][CMV][CA][CSD]
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.logging import RichHandler
from rich.console import Console
from rich.panel import Panel


# -----------------------------
# Logging
# -----------------------------
class JSONLFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created).astimezone().strftime("%Y-%m-%dT%H:%M:%S.%f%z")
        payload = {
            "ts": ts[:-8] + ts[-5:],
            "level": record.levelname,
            "name": record.name,
            "subsys": "env_registry",
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
    jsonl_handler = logging.FileHandler(logs_dir / "env_registry.jsonl", encoding="utf-8")
    jsonl_handler.setLevel(logging.DEBUG)
    jsonl_handler.setFormatter(JSONLFormatter())

    logger = logging.getLogger("utils.build_env_registry")
    logger.setLevel(logging.DEBUG)
    logger.handlers = [pretty_handler, jsonl_handler]
    logger.propagate = False

    # Enforce both handlers exist
    if not any(isinstance(h, RichHandler) for h in logger.handlers) or not any(hasattr(h, "stream") for h in logger.handlers if isinstance(h, logging.FileHandler)):
        print("✖ Logging enforcer failed: missing Pretty or JSONL handler")
        raise SystemExit(2)

    return logger


# -----------------------------
# Core
# -----------------------------
SENSITIVE_HINTS = ("TOKEN", "SECRET", "KEY", "PASSWORD", "BEARER", "API_KEY", "AUTH", "WEBHOOK")
REQUIRED_SET = {"DISCORD_TOKEN", "PROMPT_FILE", "VL_PROMPT_FILE"}

GROUP_PREFIXES = [
    "PROMETHEUS_", "X_API_", "X_", "RAG_", "STT_", "TTS_", "KOKORO_", "WEBEX_", "MEDIA_", "VIDEO_",
    "STREAMING_", "SEARCH_", "CACHE_", "OCR_", "PDF_", "SCREENSHOT_",
]


def classify_group(key: str) -> str:
    for pref in GROUP_PREFIXES:
        if key.startswith(pref):
            return pref.rstrip("_")
    return "CORE"


def classify_sensitivity(key: str) -> str:
    u = key.upper()
    return "sensitive" if any(h in u for h in SENSITIVE_HINTS) else "normal"


def choose_default(defaults: List[Optional[str]]) -> Tuple[Optional[str], List[str]]:
    # Keep literal order; filter out None/"None"
    seen: List[str] = []
    for d in defaults:
        if d is None:
            continue
        if d not in seen:
            seen.append(d)
    canonical = seen[0] if seen else None
    conflicts = seen[1:] if len(seen) > 1 else []
    return canonical, conflicts


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


def build_registry(inv: Dict[str, Any], logger: logging.Logger) -> Dict[str, List[RegistryItem]]:
    groups: Dict[str, List[RegistryItem]] = {}
    for key, meta in inv.items():
        meta.get("uses", [])
        defaults_raw = [d for d in meta.get("defaults", [])]
        coercions = [c for c in meta.get("coercions", []) if c]
        cdefault, conflicts = choose_default(defaults_raw)
        # Prefer curated type when provided, else infer from coercions
        curated_type = meta.get("type") if isinstance(meta, dict) else None
        typ = str(curated_type) if curated_type else (",".join(sorted(set(coercions))) if coercions else "unknown")

        # Prefer curated description/group/sensitivity/required when provided
        desc = str(meta.get("description") or "") if isinstance(meta, dict) else ""
        group_override = meta.get("group") if isinstance(meta, dict) else None
        group = str(group_override) if group_override else classify_group(key)

        sens_override = (meta.get("sensitivity") or "").lower() if isinstance(meta, dict) else ""
        if sens_override in ("sensitive", "secret"):
            sensitive = True
        elif sens_override in ("normal", "public"):
            sensitive = False
        else:
            sensitive = classify_sensitivity(key) == "sensitive"

        req_override = meta.get("required") if isinstance(meta, dict) else None
        required = bool(req_override) if req_override is not None else (key in REQUIRED_SET)

        item = RegistryItem(
            key=key,
            group=group,
            description=desc,
            required=required,
            sensitive=sensitive,
            type=typ,
            default=cdefault,
            all_defaults=[d for d in defaults_raw if d is not None],
            valid_values=(meta.get("valid_values") if isinstance(meta, dict) else None),
        )
        groups.setdefault(group, []).append(item)

        if conflicts:
            logger.warning(f"Default conflicts for {key}: {conflicts}")
    # Sort within groups by key
    for grp, items in groups.items():
        items.sort(key=lambda x: x.key)
    return groups


MD_HEADER = """# Canonical Environment Registry

This file is auto-generated from `utils/env_inventory.json` via `utils/build_env_registry.py`.

Fields:
- key: environment variable name
- description: to be curated (placeholder)
- required: whether startup validation requires this key
- sensitive: contains secrets (never log values)
- type: inferred from code usage (int, float, bool, list, set, unknown)
- default: canonical default when present; multiple conflicting defaults are logged
- valid_values: enumerated allowed values, if known (TBD)

[IV][CMV][SFT][CA][CDiP]
"""


def write_md(groups: Dict[str, List[RegistryItem]], out_md: Path):
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = [MD_HEADER]
    for grp in sorted(groups.keys()):
        lines.append(f"\n## {grp}")
        lines.append("")
        lines.append("| key | required | sensitive | type | default | description |")
        lines.append("|---|:---:|:---:|:---:|---|---|")
        for item in groups[grp]:
            default_disp = item.default if item.default is not None else ""
            lines.append(
                f"| {item.key} | {'✔' if item.required else ''} | {'✔' if item.sensitive else ''} | {item.type} | {default_disp} | {item.description} |"
            )
    out_md.write_text("\n".join(lines), encoding="utf-8")


def coerce_sample_value(item: RegistryItem) -> str:
    if item.sensitive:
        return ""  # leave blank in sample
    if item.default is not None:
        return item.default.strip("'")
    # Fallback based on type
    if item.type == "int":
        return "0"
    if item.type == "float":
        return "0.0"
    if item.type == "bool":
        return "false"
    return ""  # empty by default


SAMPLE_HEADER = """# Auto-generated sample environment file
# Copy to .env and edit values as needed. Sensitive values are left blank.
# Generated by utils/build_env_registry.py
"""


def write_env_sample(groups: Dict[str, List[RegistryItem]], out_env: Path):
    out_env.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = [SAMPLE_HEADER]
    for grp in sorted(groups.keys()):
        lines.append("")
        lines.append(f"# ===== {grp} =====")
        for item in groups[grp]:
            val = coerce_sample_value(item)
            lines.append(f"{item.key}={val}")
    out_env.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build canonical env registry and .env.sample")
    parser.add_argument("--inventory", default="utils/env_inventory.json", help="Path to env inventory JSON")
    parser.add_argument("--env-sample", default="configs/.env.sample", help="Output sample env file")
    parser.add_argument("--registry-md", default="docs/config/ENV_REGISTRY.md", help="Output registry markdown")
    parser.add_argument("--registry-json", default="docs/config/ENV_REGISTRY.json", help="Output registry JSON")
    args = parser.parse_args()

    logger = get_logger()
    console = Console()
    console.print(Panel.fit("Building canonical env registry", title="Env Registry"))

    inv_path = Path(args.inventory)
    if not inv_path.exists():
        logger.error(f"Inventory file not found: {inv_path}")
        return 2

    inv = json.loads(inv_path.read_text(encoding="utf-8"))
    groups = build_registry(inv, logger)

    write_env_sample(groups, Path(args.env_sample))
    write_md(groups, Path(args.registry_md))

    # Write machine-readable JSON for CI guards
    json_out = Path(args.registry_json)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        grp: [asdict(item) for item in items] for grp, items in groups.items()
    }
    json_out.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(f"✔ Wrote sample env to {args.env_sample}")
    logger.info(f"✔ Wrote registry markdown to {args.registry_md}")
    logger.info(f"✔ Wrote registry JSON to {args.registry_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
