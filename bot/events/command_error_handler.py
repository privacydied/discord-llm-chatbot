"""
Global command error handler for Discord bot.

Provides comprehensive error handling for all bot commands following
Clean Architecture (CA) and Robust Error Handling (REH) patterns.
"""
import asyncio
import logging
import traceback
from typing import Optional, Union

import discord
from discord.ext import commands
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from ..util.logging import get_logger

logger = get_logger(__name__)
console = Console()


class CommandErrorHandler:
    """
    Centralized command error handler implementing robust error patterns.
    
    Follows user rules:
    - Robust Error Handling (REH): Comprehensive error categorization
    - Clean Architecture (CA): Separation of concerns
    - Security-First Thinking (SFT): No sensitive data exposure
    """
    
    # Error categories with user-friendly messages
    ERROR_MESSAGES = {
        'command_not_found': {
            'emoji': '❓',
            'title': 'Command Not Found',
            'description': 'That command doesn\'t exist. Use `{prefix}help` to see available commands.',
            'color': discord.Color.orange()
        },
        'missing_permissions': {
            'emoji': '🔒',
            'title': 'Permission Denied',
            'description': 'You don\'t have permission to use this command.',
            'color': discord.Color.red()
        },
        'bot_missing_permissions': {
            'emoji': '🤖',
            'title': 'Bot Missing Permissions',
            'description': 'I don\'t have the required permissions to execute this command.',
            'color': discord.Color.red()
        },
        'command_on_cooldown': {
            'emoji': '⏱️',
            'title': 'Command on Cooldown',
            'description': 'This command is on cooldown. Try again in {retry_after:.1f} seconds.',
            'color': discord.Color.blue()
        },
        'bad_argument': {
            'emoji': '❌',
            'title': 'Invalid Arguments',
            'description': 'Invalid arguments provided. Check the command usage with `{prefix}help {command}`.',
            'color': discord.Color.red()
        },
        'missing_required_argument': {
            'emoji': '📝',
            'title': 'Missing Arguments',
            'description': 'Required arguments are missing. Check usage with `{prefix}help {command}`.',
            'color': discord.Color.red()
        },
        'command_invoke_error': {
            'emoji': '⚠️',
            'title': 'Command Error',
            'description': 'An error occurred while executing this command. Please try again.',
            'color': discord.Color.red()
        },
        'check_failure': {
            'emoji': '🚫',
            'title': 'Command Check Failed',
            'description': 'You don\'t meet the requirements to use this command.',
            'color': discord.Color.red()
        },
        'disabled_command': {
            'emoji': '🚧',
            'title': 'Command Disabled',
            'description': 'This command is currently disabled.',
            'color': discord.Color.orange()
        },
        'command_not_found_suggestion': {
            'emoji': '💡',
            'title': 'Did You Mean?',
            'description': 'Command not found. Did you mean: `{suggestions}`?',
            'color': discord.Color.blue()
        }
    }

    def __init__(self, bot):
        self.bot = bot
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self._command_usage_stats = {}  # Track command usage for analytics

    async def handle_command_error(self, ctx: commands.Context, error: Exception) -> None:
        """
        Main error handler dispatching to specific handlers.
        
        Args:
            ctx: Command context
            error: The exception that occurred
            
        [REH] Comprehensive error categorization and handling
        [SFT] No sensitive information exposure to users
        """
        try:
            # Update error statistics
            error_type = type(error).__name__
            self._update_error_stats(ctx, error_type)
            
            # Log error with context (following RichHandler patterns)
            await self._log_command_error(ctx, error, error_type)
            
            # Handle specific error types
            if isinstance(error, commands.CommandNotFound):
                await self._handle_command_not_found(ctx, error)
            elif isinstance(error, commands.MissingPermissions):
                await self._handle_missing_permissions(ctx, error)
            elif isinstance(error, commands.BotMissingPermissions):
                await self._handle_bot_missing_permissions(ctx, error)
            elif isinstance(error, commands.CommandOnCooldown):
                await self._handle_command_on_cooldown(ctx, error)
            elif isinstance(error, (commands.BadArgument, commands.BadUnionArgument)):
                await self._handle_bad_argument(ctx, error)
            elif isinstance(error, commands.MissingRequiredArgument):
                await self._handle_missing_required_argument(ctx, error)
            elif isinstance(error, commands.DisabledCommand):
                await self._handle_disabled_command(ctx, error)
            elif isinstance(error, commands.CheckFailure):
                await self._handle_check_failure(ctx, error)
            elif isinstance(error, commands.CommandInvokeError):
                await self._handle_command_invoke_error(ctx, error)
            else:
                await self._handle_unknown_error(ctx, error)
                
        except Exception as handler_error:
            # [REH] Error handler must never crash
            self.logger.critical(f"🚨 Error handler failed: {handler_error}", exc_info=True)
            await self._send_fallback_error_message(ctx)

    async def _log_command_error(self, ctx: commands.Context, error: Exception, error_type: str) -> None:
        """Log error with rich context information."""
        # Create structured log entry
        log_context = {
            'user_id': ctx.author.id,
            'guild_id': ctx.guild.id if ctx.guild else None,
            'channel_id': ctx.channel.id,
            'command': ctx.command.name if ctx.command else 'unknown',
            'error_type': error_type,
            'message_id': ctx.message.id if ctx.message else None
        }
        
        # Use rich logging with Tree structure for error context
        error_tree = f"""
🚨 Command Error Occurred
├── 👤 User: {ctx.author} ({ctx.author.id})
├── 🏢 Guild: {ctx.guild.name if ctx.guild else 'DM'} ({ctx.guild.id if ctx.guild else 'N/A'})
├── 📝 Channel: {ctx.channel.name if hasattr(ctx.channel, 'name') else 'DM'}
├── ⚡ Command: {ctx.command.name if ctx.command else 'unknown'}
├── 🔥 Error Type: {error_type}
└── 📋 Details: {str(error)[:100]}{'...' if len(str(error)) > 100 else ''}
        """
        
        # Log with appropriate level based on error severity
        if error_type in ['CommandNotFound']:
            self.logger.debug(error_tree.strip())
        elif error_type in ['MissingPermissions', 'CommandOnCooldown']:
            self.logger.info(error_tree.strip())
        else:
            self.logger.error(error_tree.strip(), exc_info=True)

    async def _handle_command_not_found(self, ctx: commands.Context, error: commands.CommandNotFound) -> None:
        """
        Handle command not found with smart suggestions.
        
        [REH] Convert technical error to user-friendly message
        [AS] Provide alternative suggestions when possible
        """
        # Extract attempted command name
        prefix = await self.bot.get_prefix(ctx.message)
        if isinstance(prefix, list):
            prefix = prefix[0]
            
        content = ctx.message.content
        if content.startswith(prefix):
            attempted_command = content[len(prefix):].split()[0].lower()
        else:
            attempted_command = "unknown"
        
        # Generate smart suggestions using fuzzy matching
        suggestions = await self._generate_command_suggestions(attempted_command)
        
        if suggestions:
            template = self.ERROR_MESSAGES['command_not_found_suggestion']
            embed = discord.Embed(
                title=f"{template['emoji']} {template['title']}",
                description=template['description'].format(
                    suggestions='`, `'.join(suggestions[:3])
                ),
                color=template['color']
            )
        else:
            template = self.ERROR_MESSAGES['command_not_found']
            embed = discord.Embed(
                title=f"{template['emoji']} {template['title']}",
                description=template['description'].format(prefix=prefix),
                color=template['color']
            )
        
        # Add helpful footer
        embed.set_footer(
            text="💡 Use !help to see all available commands",
            icon_url=self.bot.user.avatar.url if self.bot.user.avatar else None
        )
        
        await ctx.send(embed=embed, delete_after=30)  # Auto-delete to reduce clutter

    async def _generate_command_suggestions(self, attempted_command: str) -> list:
        """Generate smart command suggestions using fuzzy matching."""
        from difflib import get_close_matches
        
        # Get all available command names
        all_commands = []
        for command in self.bot.commands:
            all_commands.append(command.name)
            all_commands.extend(command.aliases)
        
        # Find close matches
        suggestions = get_close_matches(
            attempted_command, 
            all_commands, 
            n=3, 
            cutoff=0.6
        )
        
        return suggestions

    async def _handle_missing_permissions(self, ctx: commands.Context, error: commands.MissingPermissions) -> None:
        """Handle missing user permissions."""
        template = self.ERROR_MESSAGES['missing_permissions']
        embed = discord.Embed(
            title=f"{template['emoji']} {template['title']}",
            description=template['description'],
            color=template['color']
        )
        
        # Add specific permissions info without being too technical
        missing_perms = [perm.replace('_', ' ').title() for perm in error.missing_permissions]
        if len(missing_perms) <= 3:  # Don't overwhelm user
            embed.add_field(
                name="Required Permissions",
                value=', '.join(missing_perms),
                inline=False
            )
        
        await ctx.send(embed=embed, delete_after=30)

    async def _handle_bot_missing_permissions(self, ctx: commands.Context, error: commands.BotMissingPermissions) -> None:
        """Handle missing bot permissions."""
        template = self.ERROR_MESSAGES['bot_missing_permissions']
        embed = discord.Embed(
            title=f"{template['emoji']} {template['title']}",
            description=template['description'],
            color=template['color']
        )
        
        missing_perms = [perm.replace('_', ' ').title() for perm in error.missing_permissions]
        embed.add_field(
            name="Bot Needs These Permissions",
            value=', '.join(missing_perms[:3]),  # Limit to avoid embed size issues
            inline=False
        )
        
        await ctx.send(embed=embed)

    async def _handle_command_on_cooldown(self, ctx: commands.Context, error: commands.CommandOnCooldown) -> None:
        """Handle command cooldown with helpful timing."""
        template = self.ERROR_MESSAGES['command_on_cooldown']
        embed = discord.Embed(
            title=f"{template['emoji']} {template['title']}",
            description=template['description'].format(retry_after=error.retry_after),
            color=template['color']
        )
        
        await ctx.send(embed=embed, delete_after=min(15, error.retry_after))

    async def _handle_bad_argument(self, ctx: commands.Context, error: Union[commands.BadArgument, commands.BadUnionArgument]) -> None:
        """Handle bad arguments with usage help."""
        prefix = await self.bot.get_prefix(ctx.message)
        if isinstance(prefix, list):
            prefix = prefix[0]
            
        template = self.ERROR_MESSAGES['bad_argument']
        embed = discord.Embed(
            title=f"{template['emoji']} {template['title']}",
            description=template['description'].format(
                prefix=prefix,
                command=ctx.command.name if ctx.command else 'unknown'
            ),
            color=template['color']
        )
        
        # Add specific error details if not too technical
        error_msg = str(error)
        if len(error_msg) < 200 and not any(x in error_msg.lower() for x in ['traceback', 'exception', 'error']):
            embed.add_field(name="Issue", value=error_msg, inline=False)
        
        await ctx.send(embed=embed, delete_after=30)

    async def _handle_missing_required_argument(self, ctx: commands.Context, error: commands.MissingRequiredArgument) -> None:
        """Handle missing required arguments with parameter info."""
        prefix = await self.bot.get_prefix(ctx.message)
        if isinstance(prefix, list):
            prefix = prefix[0]
            
        template = self.ERROR_MESSAGES['missing_required_argument']
        embed = discord.Embed(
            title=f"{template['emoji']} {template['title']}",
            description=template['description'].format(
                prefix=prefix,
                command=ctx.command.name if ctx.command else 'unknown'
            ),
            color=template['color']
        )
        
        embed.add_field(
            name="Missing Parameter", 
            value=f"`{error.param.name}`",
            inline=False
        )
        
        await ctx.send(embed=embed, delete_after=30)

    async def _handle_disabled_command(self, ctx: commands.Context, error: commands.DisabledCommand) -> None:
        """Handle disabled commands."""
        template = self.ERROR_MESSAGES['disabled_command']
        embed = discord.Embed(
            title=f"{template['emoji']} {template['title']}",
            description=template['description'],
            color=template['color']
        )
        await ctx.send(embed=embed, delete_after=20)

    async def _handle_check_failure(self, ctx: commands.Context, error: commands.CheckFailure) -> None:
        """Handle check failures (custom permission checks)."""
        template = self.ERROR_MESSAGES['check_failure']
        embed = discord.Embed(
            title=f"{template['emoji']} {template['title']}",
            description=template['description'],
            color=template['color']
        )
        await ctx.send(embed=embed, delete_after=20)

    async def _handle_command_invoke_error(self, ctx: commands.Context, error: commands.CommandInvokeError) -> None:
        """
        Handle command invocation errors (internal command failures).
        
        [REH] Log technical details but show user-friendly message
        [SFT] Never expose sensitive internal information
        """
        template = self.ERROR_MESSAGES['command_invoke_error']
        embed = discord.Embed(
            title=f"{template['emoji']} {template['title']}",
            description=template['description'],
            color=template['color']
        )
        
        # Add helpful information without exposing internals
        if ctx.command:
            embed.add_field(
                name="Command", 
                value=f"`{ctx.command.name}`",
                inline=True
            )
        
        embed.set_footer(text="This error has been logged for investigation.")
        
        await ctx.send(embed=embed, delete_after=30)
        
        # Log the actual error for debugging
        self.logger.error(f"Command invoke error in {ctx.command}: {error.original}", exc_info=error.original)

    async def _handle_unknown_error(self, ctx: commands.Context, error: Exception) -> None:
        """Handle unknown/unexpected errors."""
        template = self.ERROR_MESSAGES['command_invoke_error']  # Use generic error template
        embed = discord.Embed(
            title=f"{template['emoji']} Unexpected Error",
            description="An unexpected error occurred. This has been logged for investigation.",
            color=template['color']
        )
        
        await ctx.send(embed=embed, delete_after=30)
        
        # Log with full context for debugging
        self.logger.error(f"Unknown error in command {ctx.command}: {error}", exc_info=True)

    async def _send_fallback_error_message(self, ctx: commands.Context) -> None:
        """Send ultra-simple fallback message if embed creation fails."""
        try:
            await ctx.send("❌ An error occurred. Please try again.", delete_after=10)
        except Exception:
            # If even this fails, log it but don't crash
            self.logger.critical("Failed to send fallback error message")

    def _update_error_stats(self, ctx: commands.Context, error_type: str) -> None:
        """Track error statistics for monitoring and improvement."""
        # [PA] Performance Awareness - keep stats lightweight
        key = f"{ctx.command.name if ctx.command else 'unknown'}:{error_type}"
        self._command_usage_stats[key] = self._command_usage_stats.get(key, 0) + 1
        
        # Log stats periodically for monitoring
        if sum(self._command_usage_stats.values()) % 100 == 0:  # Every 100 errors
            self.logger.info(f"📊 Error stats: {dict(list(self._command_usage_stats.items())[-5:])}")

    def get_error_statistics(self) -> dict:
        """Get error statistics for monitoring dashboard."""
        return self._command_usage_stats.copy()


async def setup_command_error_handler(bot) -> CommandErrorHandler:
    """Set up global command error handler."""
    logger = get_logger(__name__)
    
    try:
        # Create handler instance
        error_handler = CommandErrorHandler(bot)
        
        # Register the global error handler
        @bot.event
        async def on_command_error(ctx: commands.Context, error: Exception):
            """Global command error event handler."""
            await error_handler.handle_command_error(ctx, error)
        
        logger.info("✅ Global command error handler registered")
        
        # Log setup with rich formatting
        console.print(Panel(
            Text("🛡️ Command Error Handler Ready", style="bold green"),
            title="Security Enhancement",
            border_style="green"
        ))
        
        return error_handler
        
    except Exception as e:
        logger.error(f"❌ Failed to setup command error handler: {e}", exc_info=True)
        raise
