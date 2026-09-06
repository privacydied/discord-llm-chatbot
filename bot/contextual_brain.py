"""Contextual brain inference with enhanced context manager integration."""

from typing import TYPE_CHECKING, Any, Optional

import discord

from bot.brain import brain_infer
from bot.decision_helpers import resolve_scope
from bot.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from bot.core.bot import LLMBot

logger = get_logger(__name__)


async def contextual_brain_infer(
    message: discord.Message,
    prompt: str,
    bot: Optional["LLMBot"] = None,
    include_cross_user: bool = True,
    return_json_envelope: bool = False,
    *,
    perception_notes: str | None = None,
    extra_context: str | None = None,
    retrieved_context: str | None = None,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Enhanced brain inference with contextual conversation awareness.

    Args:
        message: Discord message being responded to
        prompt: The input prompt/content
        bot: Bot instance (for accessing enhanced context manager)
        include_cross_user: Whether to include cross-user context in shared channels
        return_json_envelope: Whether to return JSON envelope format
        perception_notes: Visual perception notes for the current turn
        extra_context: Local scope block (thread tail / reply chain / ambient).
            When present, the rolling history buffer is skipped: the local
            block is authoritative for that scope (ambient interjections must
            never inherit unrelated channel chatter).
        retrieved_context: RAG / curated-memory block. Unlike extra_context
            this NEVER suppresses rolling history -- retrieved knowledge
            complements recent conversation, it must not substitute for it.

    Returns:
        Dict containing response_text, used_history, and fallback status

    """
    logger.debug(f"🧠 Contextual brain inference starting [msg_id={message.id}, cross_user={include_cross_user}]")

    # Default response structure
    response_data = {"response_text": "", "used_history": [], "fallback": False}

    try:
        # Get bot instance from message if not provided
        if not bot:
            bot = message.guild.get_member(message.guild.me.id) if message.guild else None
            if not hasattr(bot, "enhanced_context_manager"):
                logger.warning("Enhanced context manager not available, falling back to basic inference")
                response_data["fallback"] = True
                response_data["response_text"] = await brain_infer(prompt, system_prompt=system_prompt)
                return response_data

        # Check if enhanced context manager is available
        if not hasattr(bot, "enhanced_context_manager") or not bot.enhanced_context_manager:
            logger.warning("Enhanced context manager not available, falling back to basic inference")
            response_data["fallback"] = True
            response_data["response_text"] = await brain_infer(prompt, system_prompt=system_prompt)
            return response_data

        # Derive include_cross_user when mention/reply context is injected (extra_context set) in non-thread replies
        include_cross_user_eff = include_cross_user
        try:
            is_thread = isinstance(message.channel, discord.Thread)
            is_reply = getattr(message, "reference", None) is not None
            if extra_context and (not is_thread) and is_reply:
                include_cross_user_eff = False
        except (AttributeError, TypeError):
            include_cross_user_eff = include_cross_user

        # Determine basic case/scope for logging and locality policy
        _scope = resolve_scope(message)
        case = _scope.case.upper() if _scope.case else "LONE"
        scope_id = _scope.scope_id

        # Skip rolling history only when an explicit LOCAL scope block is
        # present (thread tail / reply chain / ambient-local). The local block
        # is authoritative for its scope; the rolling buffer may hold
        # unrelated chatter (ambient case) or duplicate the same turns.
        # Retrieved knowledge (RAG / curated memory) must NEVER trigger this
        # skip -- it complements recent conversation, it does not replace it.
        # (A previous lone-mention carve-out compared against a scope case
        # that resolve_scope never returns; it was dead code and is removed:
        # addressed follow-ups like "the video attached" need history.)
        skip_history = bool(extra_context)

        if skip_history:
            # Optionally compute how many entries would have been dropped for visibility
            dropped = 0
            try:
                _entries = bot.enhanced_context_manager.get_context_for_user(message, include_cross_user=False)
                dropped = len(_entries or [])
            except (AttributeError, TypeError) as e:
                logger.debug(f"Failed to get context for drop count: {e}")
                dropped = 0
            # Minimal, same-schema breadcrumbs
            try:
                logger.info(
                    "scope_resolved",
                    extra={
                        "subsys": "mem.ctx",
                        "event": "scope_resolved",
                        "guild_id": getattr(getattr(message, "guild", None), "id", None),
                        "user_id": getattr(getattr(message, "author", None), "id", None),
                        "msg_id": getattr(message, "id", None),
                        "detail": {"case": case, "scope": scope_id},
                    },
                )
                logger.info(
                    "local_block",
                    extra={
                        "subsys": "mem.ctx",
                        "event": "local_block",
                        "guild_id": getattr(getattr(message, "guild", None), "id", None),
                        "user_id": getattr(getattr(message, "author", None), "id", None),
                        "msg_id": getattr(message, "id", None),
                        "detail": {"msgs": 0, "order": "local_first"},
                    },
                )
                if dropped:
                    logger.info(
                        "drop_stale",
                        extra={
                            "subsys": "mem.ctx",
                            "event": "drop_stale",
                            "guild_id": getattr(getattr(message, "guild", None), "id", None),
                            "user_id": getattr(getattr(message, "author", None), "id", None),
                            "msg_id": getattr(message, "id", None),
                            "detail": {"reason": "scope_mismatch", "dropped": dropped},
                        },
                    )
            except (AttributeError, TypeError) as e:
                logger.debug(f"Failed to log drop_stale breadcrumb: {e}")
            context_entries = []
        else:
            # Get conversation context normally
            context_entries = bot.enhanced_context_manager.get_context_for_user(message, include_cross_user=include_cross_user_eff)

        # Build contextual prompt with optional perception notes and extra context
        contextual_prompt = prompt
        history_block = None
        try:
            if context_entries:
                context_str = bot.enhanced_context_manager.format_context_string(context_entries)
                if context_str:
                    history_block = f"Conversation history:\n{context_str}"
                    logger.debug(f"✔ Context added to prompt [entries={len(context_entries)}, tokens≈{len(context_str) // 4}]")
        except (AttributeError, TypeError) as _e:
            logger.debug(f"Context history build failed: {_e}")

        perception_block = None
        if perception_notes:
            perception_block = "Perception (from the image the user replied to):\n" + perception_notes.strip()
            # Breadcrumb: show that we are injecting, but do not log full notes
            try:
                logger.info(f"🧩 Injecting perception into text prompt | chars={len(perception_notes)}")
                logger.info("Prompt preview includes: 'Perception (from the image...' section")
            except (AttributeError, TypeError) as e:
                logger.debug(f"Failed to log perception injection breadcrumb: {e}")

        blocks = []
        # Prepend mention/reply context before historical memory; retrieved
        # knowledge (RAG / curated memory) rides along without suppressing it.
        if extra_context:
            blocks.append(extra_context.strip())
        if retrieved_context and str(retrieved_context).strip():
            blocks.append(str(retrieved_context).strip())
        if history_block:
            blocks.append(history_block)
        if perception_block:
            blocks.append(perception_block)

        if blocks:
            contextual_prompt = "\n\n".join(blocks) + f"\n\nCurrent message: {prompt}"

        # Structured observability: what did the model receive for this
        # Discord message? Debug-safe metadata only, never raw content. [REH]
        try:
            _ref_id = None
            try:
                _ref = getattr(message, "reference", None)
                _ref_mid = getattr(_ref, "message_id", None) if _ref is not None else None
                _ref_id = str(_ref_mid) if _ref_mid is not None else None
            except (AttributeError, TypeError, ValueError):
                _ref_id = None
            _ref_resolved = False
            if _ref_id:
                try:
                    _ref_resolved = bot.enhanced_context_manager.get_turn_by_message_id(_ref_id) is not None
                except (AttributeError, TypeError):
                    _ref_resolved = False
            _derived_count, _derived_kinds = 0, []
            try:
                _counted = bot.enhanced_context_manager.count_derived(context_entries)
                _derived_count, _derived_kinds = int(_counted[0]), list(_counted[1])
            except (AttributeError, TypeError, ValueError, IndexError) as e:
                logger.debug(f"Derived-note telemetry unavailable: {e}")
            _guild = getattr(message, "guild", None)
            _channel = getattr(message, "channel", None)
            # Router stashes how an explicit reply was resolved
            # (discord_resolved | discord_fetch | archive | miss) in dispatch
            # metadata; surface it here so one log answers the whole query.
            _archive_lookup: bool | None = None
            try:
                _router = getattr(bot, "router", None)
                _meta = getattr(_router, "_dispatch_metadata", None)
                _entry = _meta.get(getattr(message, "id", None), {}) if isinstance(_meta, dict) else {}
                _via = _entry.get("reference_resolve") if isinstance(_entry, dict) else None
                if isinstance(_via, str):
                    _archive_lookup = _via == "archive"
            except (AttributeError, TypeError):
                _archive_lookup = None
            _retrieved = bool(retrieved_context and str(retrieved_context).strip())
            logger.info(
                "context_build",
                extra={
                    "subsys": "mem.ctx",
                    "event": "context_build",
                    "guild_id": getattr(_guild, "id", None),
                    "user_id": getattr(getattr(message, "author", None), "id", None),
                    "msg_id": getattr(message, "id", None),
                    "detail": {
                        "channel_id": getattr(_channel, "id", None),
                        "thread_id": getattr(_channel, "id", None) if isinstance(_channel, discord.Thread) else None,
                        "recent_turn_count": len(context_entries or []),
                        "history_skipped": bool(skip_history),
                        "local_block": bool(extra_context),
                        "retrieved_block": _retrieved,
                        "memory_hits": _retrieved,
                        "perception_block": bool(perception_notes),
                        "referenced_msg_id": _ref_id,
                        "referenced_msg_resolved": _ref_resolved,
                        "archive_lookup": _archive_lookup,
                        "derived_media_items": _derived_count,
                        "derived_kinds": _derived_kinds,
                    },
                },
            )
        except Exception as e:  # noqa: BLE001 - telemetry must never break the response path (mocks, odd shapes)
            # Telemetry must never break the response path (mocks, odd shapes).
            logger.debug(f"Failed to emit context_build breadcrumb: {e}")

        # Generate AI response
        ai_response = await brain_infer(contextual_prompt, system_prompt=system_prompt)

        # Get contextual response with metadata
        context_response = await bot.enhanced_context_manager.get_contextual_response(
            message=message,
            response_text=(ai_response.content if hasattr(ai_response, "content") else str(ai_response)),
            include_cross_user=include_cross_user,
        )

        # Build response data
        response_data = {
            "response_text": context_response.response_text,
            "used_history": context_response.used_history,
            "fallback": context_response.fallback,
        }

        logger.debug(f"✔ Contextual brain inference complete [fallback={context_response.fallback}]")

    except Exception as e:
        logger.error(f"❌ Contextual brain inference failed: {e}", exc_info=True)

        # Fallback to basic inference with enhanced error handling [REH]
        try:
            logger.info("🔄 Attempting fallback to basic brain inference")
            basic_response = await brain_infer(prompt, system_prompt=system_prompt)
            response_text = basic_response.content if hasattr(basic_response, "content") else str(basic_response)

            # Check if the basic response is already an error message from brain_infer
            if any(emoji in response_text for emoji in ["🤖", "🔐", "⏱️", "⏰"]):
                # brain_infer already provided a user-friendly error message
                response_data = {
                    "response_text": response_text,
                    "used_history": [],
                    "fallback": True,
                }
            else:
                # Normal fallback response
                response_data = {
                    "response_text": f"Got you — y'all wild. {response_text}",
                    "used_history": [],
                    "fallback": True,
                }
            logger.info("✅ Fallback to basic inference successful")

        except Exception as fallback_error:
            logger.exception(f"❌ Fallback inference also failed: {fallback_error}")

            # Provide user-friendly error message based on error type [REH]
            error_str = str(fallback_error).lower()
            if "no choices returned" in error_str:
                error_message = (
                    "🤖 I'm experiencing an issue with the AI service. This might be due to content filtering, API limits, or a temporary service issue. Please try rephrasing your message or try again in a moment."
                )
            elif "authentication" in error_str or "api key" in error_str:
                error_message = "🔐 There's an authentication issue with the AI service. Please contact an administrator."
            elif "rate limit" in error_str or "quota" in error_str:
                error_message = "⏱️ The AI service is currently rate-limited. Please wait a moment and try again."
            elif "timeout" in error_str:
                error_message = "⏰ The AI service timed out. Please try again with a shorter message."
            else:
                error_message = "🤖 I'm experiencing a temporary issue generating a response. Please try again in a moment."

            response_data = {
                "response_text": error_message,
                "used_history": [],
                "fallback": True,
            }
            logger.info(f"📢 Providing user-friendly error message: {error_message}")

    return response_data


async def contextual_brain_infer_simple(
    message: discord.Message,
    prompt: str,
    bot: Optional["LLMBot"] = None,
    *,
    perception_notes: str | None = None,
    extra_context: str | None = None,
    retrieved_context: str | None = None,
    system_prompt: str | None = None,
) -> str:
    """Simplified contextual brain inference that returns just the response text.
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
        retrieved_context=retrieved_context,
        system_prompt=system_prompt,
    )
    return result["response_text"]


def create_context_command_handler(bot: "LLMBot"):
    """Create command handlers for context management.

    Args:
        bot: Bot instance

    Returns:
        Dict of command handlers

    """

    async def handle_context_reset(message: discord.Message) -> str:
        """Reset conversation context."""
        if bot.enhanced_context_manager:
            await bot.enhanced_context_manager.reset_context(message)
            return "✔ Conversation context has been reset."
        return "❌ Enhanced context manager not available."

    async def handle_context_stats(message: discord.Message) -> str:
        """Get context manager statistics."""
        if bot.enhanced_context_manager:
            stats = bot.enhanced_context_manager.get_stats()
            return (
                f"📊 Context Stats:\n"
                f"• Total contexts: {stats['total_contexts']}\n"
                f"• Total messages: {stats['total_messages']}\n"
                f"• Privacy opt-outs: {stats['privacy_opt_outs']}\n"
                f"• History window: {stats['history_window']}\n"
                f"• Encryption: {'✔' if stats['encryption_enabled'] else '❌'}"
            )
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
