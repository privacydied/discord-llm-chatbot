"""
Logging helper utilities for Discord bot setup and operations.

Provides Rich-based visual logging for bot initialization, command setup,
and other operational events using Tree and Panel displays.
"""

from typing import List, Any
from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree
from rich.text import Text
import logging

logger = logging.getLogger(__name__)


def log_commands_setup(
    console: Console,
    command_modules: List[str],
    command_cogs: List[Any],
    total_commands: int,
) -> None:
    """
    Generate a Rich visual report for command setup completion.

    Args:
        console: Rich console for output
        command_modules: List of loaded command module names
        command_cogs: List of loaded command cogs
        total_commands: Total number of commands registered
    """
    try:
        # Normalize module and cog tuples into (name, success) pairs
        normalized_modules = []
        for mod in command_modules:
            if isinstance(mod, (tuple, list)) and len(mod) >= 2:
                normalized_modules.append((str(mod[0]), bool(mod[1])))
            else:
                normalized_modules.append((str(mod), True))

        normalized_cogs = []
        for cog in command_cogs:
            if isinstance(cog, (tuple, list)) and len(cog) >= 2:
                normalized_cogs.append((str(cog[0]), bool(cog[1])))
            else:
                normalized_cogs.append((str(cog), True))

        # Create command setup tree aligned with tests
        tree = Tree("🎬 [bold green]Commands Setup[/bold green]")

        modules_branch = tree.add("📦 [cyan]Import modules[/cyan]")
        for module_name, success in normalized_modules:
            status_icon = "✅" if success else "❌"
            modules_branch.add(f"{status_icon} {module_name}")

        cogs_branch = tree.add("⚙️ [yellow]Load cogs[/yellow]")
        for cog_name, success in normalized_cogs:
            status_icon = "✅" if success else "❌"
            cogs_branch.add(f"{status_icon} {cog_name}")

        loaded_count = sum(1 for _, success in normalized_modules + normalized_cogs if success)
        failed_count = len(normalized_modules + normalized_cogs) - loaded_count

        summary_branch = tree.add("📊 [magenta]Summary[/magenta]")
        summary_branch.add(f"🎉 Complete: {loaded_count} loaded, {failed_count} failed")
        summary_branch.add(f"📋 Total commands registered: {total_commands}")

        panel = Panel(tree, border_style="blue")

        console.print(panel)

        logger.info(f"✅ Command setup visualization complete: {total_commands} commands from {len(normalized_cogs)} cogs")

    except Exception as e:
        logger.error(f"❌ Failed to generate command setup report: {e}")
        # Fallback to simple logging if Rich display fails
        logger.info(f"✅ Loaded {total_commands} commands from {len(command_cogs)} cogs")


def log_startup_banner(console: Console, bot_name: str, version: str = "1.0.0") -> None:
    """
    Display a startup banner for the bot.

    Args:
        console: Rich console for output
        bot_name: Name of the bot
        version: Version string
    """
    try:
        banner_text = Text()
        banner_text.append("🤖 ", style="bold blue")
        banner_text.append(bot_name, style="bold white")
        banner_text.append(f" v{version}", style="dim white")

        panel = Panel(
            banner_text,
            title="[bold green]🚀 Bot Starting Up[/bold green]",
            border_style="green",
            padding=(1, 2),
        )

        console.print()
        console.print(panel)
        console.print()

    except Exception as e:
        logger.error(f"❌ Failed to display startup banner: {e}")


def log_shutdown_banner(console: Console, exit_code: int = 0) -> None:
    """
    Display a shutdown banner for the bot.

    Args:
        console: Rich console for output
        exit_code: Exit code (0 = graceful, >0 = error)
    """
    try:
        if exit_code == 0:
            banner_text = Text("🛑 Bot Shutdown Complete", style="bold green")
            border_style = "green"
            title = "[bold green]✅ Graceful Shutdown[/bold green]"
        else:
            banner_text = Text(f"🚨 Bot Shutdown (Exit Code: {exit_code})", style="bold red")
            border_style = "red"
            title = "[bold red]❌ Error Shutdown[/bold red]"

        panel = Panel(banner_text, title=title, border_style=border_style, padding=(1, 2))

        console.print()
        console.print(panel)
        console.print()

    except Exception as e:
        logger.error(f"❌ Failed to display shutdown banner: {e}")
