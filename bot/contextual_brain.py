"""
Contextual brain inference that integrates with the enhanced context manager
to provide conversation-aware responses with JSON envelope output.
"""

import json
from typing import Optional, Dict, Any
import discord

from bot.util.logging import get_logger
from bot.brain import brain_infer
from bot.memory.enhanced_context_manager import ContextResponse

logger = get_logger(__name__)


async def contextual_brain_infer(
    message: discord.Message,
    prompt: str,
    bot: Optional['LLMBot'] = None,
    include_cross_user: bool = True,
    return_json_envelope: bool = False,
    *,
    perception_notes: Optional[str] = None,
    extra_context: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Enhanced brain inference with contextual conversation awareness.
    
    Args:
        message: Discord message being responded to
        prompt: The input prompt/content
        bot: Bot instance (for accessing enhanced context manager)
        include_cross_user: Whether to include cross-user context in shared channels
        return_json_envelope: Whether to return JSON envelope format
        
    Returns:
        Dict containing response_text, used_history, and fallback status
    """
    logger.debug(f"🧠 Contextual brain inference starting [msg_id={message.id}, cross_user={include_cross_user}]")
    
    # Default response structure
    response_data = {
        "response_text": "",
        "used_history": [],
        "fallback": False
    }
    
    try:
        # Get bot instance from message if not provided
        if not bot:
            bot = message.guild.get_member(message.guild.me.id) if message.guild else None
            if not hasattr(bot, 'enhanced_context_manager'):
                logger.warning("Enhanced context manager not available, falling back to basic inference")
                response_data["fallback"] = True
                response_data["response_text"] = await brain_infer(prompt)
                return response_data
        
        # Check if enhanced context manager is available
        if not hasattr(bot, 'enhanced_context_manager') or not bot.enhanced_context_manager:
            logger.warning("Enhanced context manager not available, falling back to basic inference")
            response_data["fallback"] = True
            response_data["response_text"] = await brain_infer(prompt)
            return response_data
        
        # Get conversation context
        context_entries = bot.enhanced_context_manager.get_context_for_user(
            message, 
            include_cross_user=include_cross_user
        )
        
        # Build contextual prompt with optional perception notes and extra context
        contextual_prompt = prompt
        history_block = None
        try:
            if context_entries:
                context_str = bot.enhanced_context_manager.format_context_string(context_entries)
                if context_str:
                    history_block = f"Conversation history:\n{context_str}"
                    logger.debug(f"✔ Context added to prompt [entries={len(context_entries)}, tokens≈{len(context_str)//4}]")
        except Exception as _e:
            logger.debug(f"Context history build failed: {_e}")

        perception_block = None
        if perception_notes:
            perception_block = (
                "Perception (from the image the user replied to):\n" + perception_notes.strip()
            )
            # Breadcrumb: show that we are injecting, but do not log full notes
            try:
                logger.info(f"🧩 Injecting perception into text prompt | chars={len(perception_notes)}")
                logger.info("Prompt preview includes: 'Perception (from the image...' section")
            except Exception:
                pass

        blocks = []
        if history_block:
            blocks.append(history_block)
        if extra_context:
            blocks.append(extra_context.strip())
        if perception_block:
            blocks.append(perception_block)

        if blocks:
            contextual_prompt = "\n\n".join(blocks) + f"\n\nCurrent message: {prompt}"
        
        # Generate AI response
        ai_response = await brain_infer(contextual_prompt)
        
        # Get contextual response with metadata
        context_response = await bot.enhanced_context_manager.get_contextual_response(
            message=message,
            response_text=ai_response.content if hasattr(ai_response, 'content') else str(ai_response),
            include_cross_user=include_cross_user
        )
        
        # Build response data
        response_data = {
            "response_text": context_response.response_text,
            "used_history": context_response.used_history,
            "fallback": context_response.fallback
        }
        
        logger.debug(f"✔ Contextual brain inference complete [fallback={context_response.fallback}]")
        
    except Exception as e:
        logger.error(f"❌ Contextual brain inference failed: {e}", exc_info=True)
        # Fallback to basic inference
        try:
            basic_response = await brain_infer(prompt)
            response_data = {
                "response_text": f"Got you — y'all wild. {basic_response.content if hasattr(basic_response, 'content') else str(basic_response)}",
                "used_history": [],
                "fallback": True
            }
        except Exception as fallback_error:
            logger.error(f"❌ Fallback inference also failed: {fallback_error}")
            response_data = {
                "response_text": "⚠️ I'm having trouble processing your message right now. Please try again.",
                "used_history": [],
                "fallback": True
            }
    
    return response_data


async def contextual_brain_infer_simple(
    message: discord.Message,
    prompt: str,
    bot: Optional['LLMBot'] = None,
    *,
    perception_notes: Optional[str] = None,
    extra_context: Optional[str] = None,
) -> str:
    """
    Simplified contextual brain inference that returns just the response text.
    For backward compatibility with existing code.
    
    Args:
        message: Discord message being responded to
        prompt: The input prompt/content
        bot: Bot instance
        
    Returns:
        Response text string
    """
    result = await contextual_brain_infer(
        message,
        prompt,
        bot,
        return_json_envelope=False,
        perception_notes=perception_notes,
        extra_context=extra_context,
    )
    return result["response_text"]


def create_context_command_handler(bot: 'LLMBot'):
    """
    Create command handlers for context management.
    
    Args:
        bot: Bot instance
        
    Returns:
        Dict of command handlers
    """
    
    async def handle_context_reset(message: discord.Message) -> str:
        """Reset conversation context."""
        if bot.enhanced_context_manager:
            bot.enhanced_context_manager.reset_context(message)
            return "✔ Conversation context has been reset."
        return "❌ Enhanced context manager not available."
    
    async def handle_context_stats(message: discord.Message) -> str:
        """Get context manager statistics."""
        if bot.enhanced_context_manager:
            stats = bot.enhanced_context_manager.get_stats()
            return f"📊 Context Stats:\n" \
                   f"• Total contexts: {stats['total_contexts']}\n" \
                   f"• Total messages: {stats['total_messages']}\n" \
                   f"• Privacy opt-outs: {stats['privacy_opt_outs']}\n" \
                   f"• History window: {stats['history_window']}\n" \
                   f"• Encryption: {'✔' if stats['encryption_enabled'] else '❌'}"
        return "❌ Enhanced context manager not available."
    
    async def handle_privacy_optout(message: discord.Message, opt_out: bool = True) -> str:
        """Handle privacy opt-out/opt-in."""
        if bot.enhanced_context_manager:
            bot.enhanced_context_manager.set_privacy_opt_out(message.author.id, opt_out)
            action = "opted out of" if opt_out else "opted into"
            return f"✔ You have been {action} conversation context tracking."
        return "❌ Enhanced context manager not available."
    
    return {
        "context_reset": handle_context_reset,
        "context_stats": handle_context_stats,
        "privacy_optout": lambda msg: handle_privacy_optout(msg, True),
        "privacy_optin": lambda msg: handle_privacy_optout(msg, False),
    }
