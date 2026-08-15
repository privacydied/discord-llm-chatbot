"""Safety invariants for the model-callable tool layer.
[SFT][IV].

These tests exist to keep a promise: the bot cannot delete files on the host
or run shell commands, and cannot acquire that power by accident later.

They fail the build if anyone adds a dangerous import or call anywhere under
bot/tools/, or widens the allowlist to something that smells like code
execution or filesystem mutation. Deleting or weakening these tests should be
treated as a security change, not a test cleanup.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

TOOLS_DIR = Path("bot/tools")

# Modules that grant process or filesystem control. None of them has any place
# in a tool the model can invoke. [SFT]
FORBIDDEN_IMPORTS: frozenset[str] = frozenset(
    {
        "subprocess",
        "shutil",
        "pty",
        "ctypes",
        "multiprocessing",
        "socket",
        "shlex",
        "commands",
        "popen2",
        "fcntl",
        "resource",
        "signal",
    }
)

# Builtins that turn data into code, or open the filesystem. [SFT]
FORBIDDEN_CALLS: frozenset[str] = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "input",
        "breakpoint",
    }
)

# Attribute calls that mutate the filesystem or spawn processes. [SFT]
FORBIDDEN_ATTR_CALLS: frozenset[str] = frozenset(
    {
        "system",
        "popen",
        "remove",
        "unlink",
        "rmdir",
        "rmtree",
        "spawn",
        "spawnl",
        "spawnv",
        "fork",
        "execv",
        "execve",
        "kill",
        "chmod",
        "chown",
        "rename",
        "replace",
        "truncate",
        "write_text",
        "write_bytes",
        "mkdir",
        "makedirs",
        "touch",
    }
)

# Substrings that must never appear in an allowlisted tool name -- a tripwire
# for someone adding a shell or file tool to ALLOWED_TOOL_NAMES. [SFT]
FORBIDDEN_NAME_FRAGMENTS: tuple[str, ...] = (
    "shell",
    "exec",
    "eval",
    "bash",
    "sh_",
    "command",
    "cmd",
    "delete",
    "remove",
    "unlink",
    "rm_",
    "file",
    "path",
    "write",
    "spawn",
    "process",
    "subprocess",
    "os_",
    "system",
    "sudo",
    "chmod",
)


def _tool_source_files() -> list[Path]:
    files = sorted(TOOLS_DIR.rglob("*.py"))
    assert files, "no source files found under bot/tools — has the package moved?"
    return files


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


@pytest.mark.parametrize("path", _tool_source_files(), ids=lambda p: str(p))
def test_no_dangerous_imports(path: Path):
    """No tool module may import a process- or filesystem-control module."""
    tree = _parse(path)
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            offenders += [a.name.split(".")[0] for a in node.names if a.name.split(".")[0] in FORBIDDEN_IMPORTS]
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            if root in FORBIDDEN_IMPORTS:
                offenders.append(root)
    assert not offenders, f"{path} imports forbidden module(s): {sorted(set(offenders))}"


@pytest.mark.parametrize("path", _tool_source_files(), ids=lambda p: str(p))
def test_no_code_execution_or_file_opening(path: Path):
    """No tool module may call eval/exec/compile/__import__/open."""
    tree = _parse(path)
    offenders = [node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS]
    assert not offenders, f"{path} calls forbidden builtin(s): {sorted(set(offenders))}"


@pytest.mark.parametrize("path", _tool_source_files(), ids=lambda p: str(p))
def test_no_filesystem_or_process_mutation(path: Path):
    """No tool module may call a filesystem-mutating or process-spawning method."""
    tree = _parse(path)
    offenders = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in FORBIDDEN_ATTR_CALLS]
    assert not offenders, f"{path} calls forbidden method(s): {sorted(set(offenders))}"


def test_allowlist_names_are_read_only_sounding():
    """Tripwire: an allowlisted name must not suggest code or file access."""
    from bot.tools.registry import ALLOWED_TOOL_NAMES

    for name in ALLOWED_TOOL_NAMES:
        lowered = name.lower()
        hits = [frag for frag in FORBIDDEN_NAME_FRAGMENTS if frag in lowered]
        assert not hits, f"allowlisted tool {name!r} contains suspicious fragment(s) {hits}; if this is genuinely safe, review it deliberately and adjust this test"


def test_registry_exposes_exactly_the_allowlist():
    """Registered tools and the allowlist must agree — no drift in either direction."""
    from bot.tools.registry import ALLOWED_TOOL_NAMES, get_registry

    assert set(get_registry().names()) == set(ALLOWED_TOOL_NAMES)


def test_cannot_register_outside_allowlist():
    """Registration is the gate; it must reject an unlisted name."""
    from bot.tools.registry import ToolRegistrationError, ToolRegistry
    from bot.tools.types import ToolResult, ToolSpec

    async def _handler(ctx, args):
        return ToolResult.success("nope")

    registry = ToolRegistry()
    spec = ToolSpec(name="run_shell_command", description="d", parameters={}, handler=_handler)
    with pytest.raises(ToolRegistrationError):
        registry.register(spec)
    assert registry.names() == []


def test_duplicate_registration_rejected():
    from bot.tools.builtins.clock import SPEC
    from bot.tools.registry import ToolRegistrationError, ToolRegistry

    registry = ToolRegistry()
    registry.register(SPEC)
    with pytest.raises(ToolRegistrationError):
        registry.register(SPEC)


async def test_unknown_tool_name_is_a_miss_not_a_dispatch():
    """An invented name must fail closed, reaching no interpreter of any kind."""
    from bot.tools import ToolContext, execute_tool

    for name in ("rm", "os.system", "subprocess.run", "../../etc/passwd", "eval"):
        result = await execute_tool(name, {}, ToolContext())
        assert not result.ok
        assert "unknown tool" in (result.error or "")


async def test_non_dict_arguments_rejected():
    from bot.tools import ToolContext, execute_tool

    result = await execute_tool("get_current_time", "not a dict", ToolContext())
    assert not result.ok


def test_builtin_specs_are_explicitly_listed():
    """Tools come from a hand-written tuple, not a directory scan."""
    from bot.tools.builtins import BUILTIN_SPECS
    from bot.tools.registry import ALLOWED_TOOL_NAMES

    assert {spec.name for spec in BUILTIN_SPECS} == set(ALLOWED_TOOL_NAMES)


def test_tool_context_exposes_no_filesystem_handle():
    """ToolContext must not hand tools a path, file or subprocess facility."""
    from bot.tools.types import ToolContext

    ctx = ToolContext()
    public = {f for f in dir(ctx) if not f.startswith("_")}
    assert public <= {"message", "bot", "config", "channel"}, f"ToolContext gained unexpected surface: {public}"
