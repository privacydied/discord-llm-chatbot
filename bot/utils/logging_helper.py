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
    total_commands: int
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
        # Create command setup tree
        tree = Tree("🚀 [bold green]Command Extensions Loaded[/bold green]")
        
        # Add modules branch
        modules_branch = tree.add("📦 [cyan]Modules[/cyan]")
        for module in command_modules:
            module_name = module.split('.')[-1] if '.' in module else module
            modules_branch.add(f"✅ {module_name}")
        
        # Add cogs branch with command counts
        cogs_branch = tree.add("🔧 [yellow]Cogs[/yellow]")
        for cog in command_cogs:
            try:
                # Handle both cog objects and potential tuples/other structures
                if hasattr(cog, '__class__') and hasattr(cog, 'get_commands'):
                    cog_name = cog.__class__.__name__
                    cog_commands = len([cmd for cmd in cog.get_commands() if not cmd.hidden])
                    cogs_branch.add(f"✅ {cog_name} ({cog_commands} commands)")
                else:
                    # Fallback for unexpected data structures
                    cog_name = str(cog) if hasattr(cog, '__str__') else repr(cog)
                    cogs_branch.add(f"✅ {cog_name}")
            except Exception as e:
                logger.debug(f"Skipping malformed cog entry: {e}")
                continue
        
        # Add summary
        summary_branch = tree.add("📊 [magenta]Summary[/magenta]")
        summary_branch.add(f"Total Commands: [bold]{total_commands}[/bold]")
        summary_branch.add(f"Loaded Cogs: [bold]{len(command_cogs)}[/bold]")
        summary_branch.add(f"Modules: [bold]{len(command_modules)}[/bold]")
        
        # Create panel with tree
        panel = Panel(
            tree,
            title="[bold blue]🎯 Bot Command Setup Complete[/bold blue]",
            border_style="blue",
            padding=(1, 2)
        )
        
        console.print()
        console.print(panel)
        console.print()
        
        logger.info(f"✅ Command setup visualization complete: {total_commands} commands from {len(command_cogs)} cogs")
        
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
            padding=(1, 2)
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
        
        panel = Panel(
            banner_text,
            title=title,
            border_style=border_style,
            padding=(1, 2)
        )
        
        console.print()
        console.print(panel)
        console.print()
        
    except Exception as e:
        logger.error(f"❌ Failed to display shutdown banner: {e}")
