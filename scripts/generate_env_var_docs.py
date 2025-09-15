#!/usr/bin/env python3
"""Generate docs/ENV_VARS.md by scanning the codebase for environment variables."""

import re
import pathlib

ENV_EXAMPLE_PATH = pathlib.Path(".env.example")


def heuristic_description(name: str) -> str:
    words = name.lower().split("_")
    if "enable" in words or "enabled" in words:
        target = " ".join(w for w in words if w not in {"enable", "enabled"})
        target = target.strip() or "feature"
        return f"Enable or disable {target}."
    if "path" in words:
        target = " ".join(w for w in words if w != "path")
        return f"Path to {target}."
    if "dir" in words or "directory" in words:
        target = " ".join(w for w in words if w not in {"dir", "directory"})
        return f"Directory for {target}."
    if "timeout" in words:
        unit = "ms" if "ms" in words else "seconds"
        target = " ".join(w for w in words if w not in {"timeout", "ms", "s"})
        target = target.replace("_", " ").strip() or "operation"
        return f"Timeout for {target} in {unit}."
    if words and words[0] == "max":
        return f"Maximum {' '.join(words[1:])}."
    if words and words[0] == "min":
        return f"Minimum {' '.join(words[1:])}."
    if "port" in words:
        target = " ".join(w for w in words if w != "port")
        return f"Port for {target}."
    if "url" in words:
        target = " ".join(w for w in words if w != "url")
        return f"URL for {target}."
    if "api" in words:
        target = " ".join(w for w in words if w != "api")
        return f"API setting for {target}."
    return f"Configuration for {' '.join(words)}."


def parse_env_example() -> dict[str, tuple[str, str]]:
    env_map: dict[str, tuple[str, str]] = {}
    if not ENV_EXAMPLE_PATH.exists():
        return env_map
    prev_comments: list[str] = []
    for raw in ENV_EXAMPLE_PATH.read_text().splitlines():
        line = raw.strip()
        if not line:
            prev_comments = []
            continue
        if line.startswith("#"):
            comment = line.lstrip("#").strip()
            if comment.startswith("=====") and comment.endswith("====="):
                prev_comments = []
            elif set(comment) <= {"=", "-"}:
                prev_comments = []
            elif len(comment.split()) <= 3 and not comment.endswith(
                "."
            ):  # likely a section heading
                prev_comments = []
            else:
                prev_comments.append(comment)
            continue
        if "=" not in line:
            continue
        name, rest = line.split("=", 1)
        name = name.strip()
        value = rest.strip()
        desc = ""
        if "#" in value:
            value, comment = value.split("#", 1)
            value = value.strip()
            desc = comment.strip()
        elif prev_comments:
            desc = prev_comments[-1]
        env_map[name] = (value, desc or heuristic_description(name))
        prev_comments = []
    return env_map


def collect_env_vars() -> dict[str, tuple[str, str]]:
    patterns = [
        re.compile(r"os\.getenv\(['\"]([A-Z0-9_]+)['\"](?:,\s*([^\)]+))?\)"),
        re.compile(r"os\.environ\.get\(['\"]([A-Z0-9_]+)['\"](?:,\s*([^\)]+))?\)"),
        re.compile(r"os\.environ\[['\"]([A-Z0-9_]+)['\"]\]"),
    ]
    files = list(pathlib.Path("bot").rglob("*.py")) + list(
        pathlib.Path(".").glob("*.py")
    )
    vars: dict[str, tuple[str, str]] = parse_env_example()
    for path in files:
        text = path.read_text(errors="ignore")
        for name, default in patterns[0].findall(text) + patterns[1].findall(text):
            default = default.strip() if default else ""
            if default.startswith(("'", '"')) and default.endswith(("'", '"')):
                default = default[1:-1]
            if "(" in default or "[" in default or "os." in default:
                default = ""
            if name not in vars:
                vars[name] = (default, heuristic_description(name))
        for name in patterns[2].findall(text):
            if name not in vars:
                vars[name] = ("", heuristic_description(name))
    return vars


def generate_markdown(vars: dict[str, tuple[str, str]]) -> str:
    lines = [
        "# Environment Variables (auto-generated)",
        "",
        "| Name | Default | Description |",
        "| --- | --- | --- |",
    ]
    for name in sorted(vars):
        default, desc = vars[name]
        default = default or "—"
        desc = desc or heuristic_description(name)
        lines.append(f"| `{name}` | `{default}` | {desc} |")
    return "\n".join(lines)


def main() -> None:
    vars = collect_env_vars()
    markdown = generate_markdown(vars)
    pathlib.Path("docs/ENV_VARS.md").write_text(markdown)


if __name__ == "__main__":
    main()
