"""
Image upgrade commands for Twitter/X image-only tweets.
Provides emoji-driven upgrade functionality: detailed descriptions, OCR, tags, explanations, and thread context.
"""

import asyncio
from typing import Dict, List, Optional, Any
import discord
from discord.ext import commands

from ..utils.logging import get_logger

logger = get_logger(__name__)


class ImageUpgradeManager:
    """
    Manages emoji-driven upgrades for image-only tweet responses.
    Stores upgrade context and handles reaction-based expansions. [CA][REH]
    """

    def __init__(self, bot):
        self.bot = bot
        self.config = bot.config
        self.upgrade_cache: Dict[int, Dict[str, Any]] = {}  # message_id -> upgrade data
        self.cache_ttl = 3600  # 1 hour TTL for upgrade context

        # Parse enabled upgrade reactions from config
        reactions_str = self.config.get("IMAGE_UPGRADE_REACTIONS", "🖼️,🔎,🏷️,🧠,↩️")
        self.enabled_reactions = [
            r.strip() for r in reactions_str.split(",") if r.strip()
        ]

        logger.info(
            f"✅ ImageUpgradeManager initialized with reactions: {self.enabled_reactions}"
        )

    def store_upgrade_context(
        self,
        message_id: int,
        url_or_context: Dict[str, Any] | str,
        syn_data: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None,
        original_analysis: Optional[List[str]] = None,
    ):
        """
        Store upgrade context for a message to enable reaction-based expansions.

        Supports both the expanded signature (message_id, url, syn_data, source,
        original_analysis) and a compact form where a context dict is provided.
        """
        if isinstance(url_or_context, dict) and syn_data is None:
            context = url_or_context
            url = context.get("url")
            syn_data = context.get("syndication_data") or context.get("syn_data") or {}
            source = context.get("source") or "unknown"
            original_analysis = context.get("original_analysis") or []
        else:
            url = str(url_or_context)
            syn_data = syn_data or {}
            source = source or "unknown"
            original_analysis = original_analysis or []

        self.upgrade_cache[message_id] = {
            "url": url,
            "syn_data": syn_data,
            "source": source,
            "original_analysis": original_analysis,
            "timestamp": asyncio.get_event_loop().time(),
            "upgraded": set(),  # Track which upgrades were already applied
        }

        logger.info(
            f"📝 Stored upgrade context for message {message_id}: {len(original_analysis)} images"
        )

        # Return an awaitable-friendly object so tests can optionally await the call
        fut = asyncio.get_event_loop().create_future()
        fut.set_result(None)
        return fut

    def get_upgrade_context(self, message_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve cached upgrade context if still valid."""
        context = self.upgrade_cache.get(message_id)
        if not context:
            return None

        current_time = asyncio.get_event_loop().time()
        if current_time - context.get("timestamp", 0) > self.cache_ttl:
            self.upgrade_cache.pop(message_id, None)
            return None

        return context

    async def add_upgrade_reactions(self, message: discord.Message) -> None:
        """Add emoji reactions for available upgrades to a message. [REH]"""
        try:
            for emoji in self.enabled_reactions:
                await message.add_reaction(emoji)
                await asyncio.sleep(0.2)  # Rate limit protection

            logger.info(
                f"✅ Added {len(self.enabled_reactions)} upgrade reactions to message {message.id}"
            )

        except Exception as e:
            logger.error(
                f"❌ Failed to add upgrade reactions to message {message.id}: {e}"
            )

    async def handle_upgrade_reaction(
        self, payload: discord.RawReactionActionEvent
    ) -> Optional[str]:
        """
        Handle upgrade reaction and return expanded content.

        Args:
            payload: Discord reaction event payload

        Returns:
            Expanded content string or None if no upgrade needed
        """
        try:
            message_id = payload.message_id
            emoji = getattr(payload.emoji, "name", str(payload.emoji))
            user_id = payload.user_id

            # Skip bot's own reactions
            if user_id == self.bot.user.id:
                return None

            # Check if we have upgrade context for this message
            context = self.get_upgrade_context(message_id)
            if not context:
                logger.debug(f"🔍 No upgrade context found for message {message_id}")
                return None

            # Check if this emoji is enabled and not already applied
            if emoji not in self.enabled_reactions:
                return None

            upgrade_key = f"{emoji}_{user_id}"
            if upgrade_key in context["upgraded"]:
                logger.debug(
                    f"🔄 Upgrade {emoji} already applied by user {user_id} for message {message_id}"
                )
                return None

            # Process the upgrade
            expanded_content = await self._process_upgrade(emoji, context)

            if expanded_content:
                # Mark this upgrade as applied
                context["upgraded"].add(upgrade_key)
                logger.info(
                    f"✅ Applied upgrade {emoji} for user {user_id} on message {message_id}"
                )
                return expanded_content

            return None

        except Exception as e:
            logger.error(f"❌ Error handling upgrade reaction: {e}", exc_info=True)
            return None

    async def _process_upgrade(
        self, emoji: str, context: Dict[str, Any]
    ) -> Optional[str]:
        """
        Process a specific upgrade request and return expanded content.

        Args:
            emoji: Upgrade emoji that was reacted
            context: Cached upgrade context

        Returns:
            Expanded content string or None
        """
        try:
            url = context["url"]
            syn_data = context["syn_data"]
            context["source"]
            original_analysis = context["original_analysis"]
            photos = syn_data.get("photos") or []

            if emoji == "🖼️":  # Detailed caption
                return await self._generate_detailed_caption(
                    photos, original_analysis, url
                )

            elif emoji == "🔎":  # OCR details
                return await self._generate_ocr_details(photos, url)

            elif emoji == "🏷️":  # Tags
                return await self._generate_tags(original_analysis, url)

            elif emoji == "🧠":  # Explain
                return await self._generate_explanation(
                    syn_data, original_analysis, url
                )

            elif emoji == "↩️":  # Thread context
                return await self._generate_thread_context(syn_data, url)

            else:
                logger.warning(f"⚠️ Unknown upgrade emoji: {emoji}")
                return None

        except Exception as e:
            logger.error(f"❌ Error processing upgrade {emoji}: {e}", exc_info=True)
            return None

    async def _get_detailed_vision_analysis(
        self, photo: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Stub for detailed vision analysis; can be patched in tests."""
        return None

    async def _get_ocr_analysis(self, photo: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Stub for OCR extraction; can be patched in tests."""
        return None

    async def _get_thread_context(
        self, syn_data: Dict[str, Any], url: str
    ) -> Optional[Dict[str, Any]]:
        """Stub for thread context retrieval; can be patched in tests."""
        return None

    async def _generate_detailed_caption(
        self, photos: List[Dict[str, Any]], original_analysis: List[str], url: str
    ) -> str:
        """Generate detailed image captions with composition and attributes. [CA]"""
        try:
            detailed_parts = ["🖼️ **Detailed Caption**"]

            for idx, photo in enumerate(photos, start=1):
                analysis_list = original_analysis or []
                analysis = analysis_list[idx - 1] if idx - 1 < len(analysis_list) else ""
                photo_url = (
                    photo.get("url") or photo.get("image_url") or photo.get("src")
                )
                if not photo_url:
                    continue

                detailed_analysis = await self._get_detailed_vision_analysis(photo)
                if detailed_analysis is None:
                    router = self.bot.router if hasattr(self.bot, "router") else None
                    if router and hasattr(router, "_vl_describe_image_from_url"):
                        detailed_analysis = await router._vl_describe_image_from_url(
                            photo_url, prompt="Provide a detailed description"
                        )

                text_block = ""
                if isinstance(detailed_analysis, dict):
                    text_block = detailed_analysis.get("alt_text") or ""
                    if detailed_analysis.get("ocr_text"):
                        text_block += f"\n{detailed_analysis['ocr_text']}"
                elif detailed_analysis:
                    text_block = str(detailed_analysis)
                else:
                    text_block = analysis or ""

                prefix = (
                    f"**Image {idx}/{len(photos)}:** " if len(photos) > 1 else ""
                )
                detailed_parts.append(f"{prefix}{text_block}".strip())

            return "\n\n".join([p for p in detailed_parts if p])

        except Exception as e:
            logger.error(f"❌ Error generating detailed caption: {e}")
            return "⚠️ Could not generate detailed description right now."

    async def _generate_ocr_details(
        self, photos: List[Dict[str, Any]], url: str
    ) -> str:
        """Generate detailed OCR text extraction from images. [IV]"""
        try:
            ocr_parts = ["🔎 **OCR Text Details**"]
            found_text = False

            for idx, photo in enumerate(photos, start=1):
                photo_url = (
                    photo.get("url") or photo.get("image_url") or photo.get("src")
                )
                if not photo_url:
                    continue

                # OCR-focused prompt
                ocr_prompt = (
                    "Extract and transcribe ALL visible text from this image. Include text on signs, "
                    "labels, documents, screens, clothing, or any other objects. Preserve formatting "
                    "where possible and indicate if text is partially obscured or unclear."
                )

                try:
                    router = self.bot.router if hasattr(self.bot, "router") else None
                    ocr_result = await self._get_ocr_analysis(photo)
                    if ocr_result is None and router and hasattr(
                        router, "_vl_describe_image_from_url"
                    ):
                        ocr_result = await router._vl_describe_image_from_url(
                            photo_url, prompt=ocr_prompt
                        )

                    if isinstance(ocr_result, dict):
                        ocr_text = ocr_result.get("ocr_text") or ""
                    else:
                        ocr_text = ocr_result or ""

                    if ocr_text and any(
                        keyword in ocr_text.lower()
                        for keyword in ["text", "says", "reads", '"', "invoice"]
                    ):
                        found_text = True
                        header = (
                            f"**Image {idx}/{len(photos)}:**"
                            if len(photos) > 1
                            else ""
                        )
                        ocr_parts.append(f"{header}\n{ocr_text}".strip())
                    else:
                        header = (
                            f"**Image {idx}/{len(photos)}:**"
                            if len(photos) > 1
                            else ""
                        )
                        ocr_parts.append(
                            f"{header}\n*No readable text detected*".strip()
                        )

                except Exception as ocr_err:
                    logger.error(f"❌ OCR analysis failed for image {idx}: {ocr_err}")
                    ocr_parts.append(
                        f"**Image {idx}/{len(photos)}:** Could not extract text"
                    )

            if not found_text and len(photos) == 1:
                return "**🔎 Text Content (OCR)**\n\n*No readable text detected in this image.*"

            return "\n\n".join(ocr_parts)

        except Exception as e:
            logger.error(f"❌ Error generating OCR details: {e}")
            return "⚠️ Could not extract text content right now."

    async def _generate_tags(self, original_analysis: List[str], url: str) -> str:
        """Generate keyword tags for searchability. [AS]"""
        try:
            if not bool(self.config.get("VISION_TAGS_ENABLE", True)):
                return "**🏷️ Tags:** *Tag generation is disabled*"

            # Simple tag extraction from analysis
            all_text = " ".join(original_analysis)

            # Basic keyword extraction (this would be more sophisticated in production)

            # Extract potential entities and objects
            common_objects = [
                "person",
                "people",
                "man",
                "woman",
                "child",
                "car",
                "building",
                "tree",
                "food",
                "animal",
                "dog",
                "cat",
                "book",
                "phone",
                "computer",
                "sign",
                "text",
                "outdoor",
                "indoor",
                "street",
                "room",
                "kitchen",
                "office",
            ]

            found_tags = []
            text_lower = all_text.lower()

            for obj in common_objects:
                if obj in text_lower:
                    found_tags.append(obj)

            # Add some context tags
            if (
                "outdoor" in text_lower
                or "street" in text_lower
                or "building" in text_lower
            ):
                found_tags.append("outdoor")
            if "indoor" in text_lower or "room" in text_lower or "inside" in text_lower:
                found_tags.append("indoor")

            # Remove duplicates and limit
            found_tags = list(set(found_tags))[:10]

            if found_tags:
                tags_str = " • ".join(found_tags)
                return f"**🏷️ Tags**\n\n{tags_str}"
            else:
                return "**🏷️ Tags**\n\n*No specific tags identified*"

        except Exception as e:
            logger.error(f"❌ Error generating tags: {e}")
            return "⚠️ Could not generate tags right now."

    async def _generate_explanation(
        self, syn_data: Dict[str, Any], original_analysis: List[str], url: str
    ) -> str:
        """Generate neutral explanation of why the image might be interesting. [SFT]"""
        try:
            # Extract context
            user = syn_data.get("user") or {}
            username = user.get("screen_name") or "unknown"

            explanation_parts = ["**🧠 Context & Interest**"]

            # Analyze why this might be noteworthy (neutral tone)
            analysis_text = " ".join(original_analysis)

            # Basic interest detection patterns
            interest_indicators = []

            if any(
                word in analysis_text.lower()
                for word in ["unusual", "unique", "interesting", "notable"]
            ):
                interest_indicators.append("Contains noteworthy visual elements")

            if any(
                word in analysis_text.lower()
                for word in ["text", "sign", "writing", "document"]
            ):
                interest_indicators.append("Contains readable text or signage")

            if any(
                word in analysis_text.lower() for word in ["person", "people", "group"]
            ):
                interest_indicators.append("Shows people or social activity")

            if any(
                word in analysis_text.lower() for word in ["product", "brand", "logo"]
            ):
                interest_indicators.append("Features commercial or branded content")

            # Build neutral explanation
            if interest_indicators:
                explanation_parts.append("This image appears notable because:")
                for indicator in interest_indicators[:3]:  # Limit to top 3
                    explanation_parts.append(f"• {indicator}")
            else:
                explanation_parts.append(
                    "This appears to be a general social media image without specific notable features."
                )

            explanation_parts.append(
                f"\n*Shared by @{username} without accompanying text.*"
            )

            return "\n".join(explanation_parts)

        except Exception as e:
            logger.error(f"❌ Error generating explanation: {e}")
            return "⚠️ Could not generate explanation right now."

    async def _generate_thread_context(self, syn_data: Dict[str, Any], url: str) -> str:
        """Generate thread context if this tweet is part of a conversation. [AS]"""
        try:
            override = await self._get_thread_context(syn_data, url)
            if override:
                if isinstance(override, str):
                    return override
                if isinstance(override, dict):
                    tweets = override.get("thread_tweets") or []
                    parts = ["↩️ **Thread Context**"]
                    for tweet in tweets:
                        user = tweet.get("user") or "unknown"
                        text = tweet.get("text") or ""
                        parts.append(f"- @{user}: {text}")
                    return "\n".join(parts)

            # This would integrate with Twitter API to get conversation context
            # For now, provide basic tweet metadata

            context_parts = ["↩️ **Thread Context**"]

            user = syn_data.get("user") or {}
            username = user.get("screen_name") or "unknown"
            created_at = syn_data.get("created_at")

            context_parts.append(f"**Author:** @{username}")
            if created_at:
                context_parts.append(f"**Posted:** {created_at}")

            # Check for reply indicators in syndication data
            if syn_data.get("in_reply_to_status_id"):
                context_parts.append("**Type:** Reply to another tweet")
            elif syn_data.get("is_quote_status"):
                context_parts.append("**Type:** Quote tweet")
            else:
                context_parts.append("**Type:** Original tweet")

            context_parts.append(f"**URL:** {url}")

            # Note about thread context limitation
            context_parts.append(
                "\n*Thread context limited - this shows basic tweet metadata only.*"
            )

            return "\n".join(context_parts)

        except Exception as e:
            logger.error(f"❌ Error generating thread context: {e}")
            return "⚠️ Could not generate thread context right now."

    async def cleanup_expired_cache(self) -> None:
        """Clean up expired upgrade contexts. [RM]"""
        try:
            current_time = asyncio.get_event_loop().time()
            expired_keys = []

            for message_id, context in self.upgrade_cache.items():
                if current_time - context["timestamp"] > self.cache_ttl:
                    expired_keys.append(message_id)

            for key in expired_keys:
                del self.upgrade_cache[key]

            if expired_keys:
                logger.info(
                    f"🧹 Cleaned up {len(expired_keys)} expired upgrade contexts"
                )

        except Exception as e:
            logger.error(f"❌ Error cleaning up upgrade cache: {e}")


class ImageUpgradeCommands(commands.Cog):
    """Discord cog for handling image upgrade reactions and commands."""

    def __init__(self, bot):
        self.bot = bot
        self.upgrade_manager = ImageUpgradeManager(bot)
        logger.info("✅ ImageUpgradeCommands cog initialized")

    @commands.Cog.listener()
    async def on_raw_reaction_add(self, payload: discord.RawReactionActionEvent):
        """Handle reaction additions for image upgrades."""
        try:
            # Only process reactions in guilds and DMs where bot is active
            if payload.guild_id is None and not isinstance(payload.channel_id, int):
                return

            # Skip if bot added the reaction
            if payload.user_id == self.bot.user.id:
                return

            # Check for upgrade reaction
            expanded_content = await self.upgrade_manager.handle_upgrade_reaction(
                payload
            )

            if expanded_content:
                # Get the channel and message
                channel = self.bot.get_channel(payload.channel_id)
                if not channel:
                    return

                try:
                    message = await channel.fetch_message(payload.message_id)

                    # Edit the original message to include the upgrade
                    current_content = message.content or ""

                    # Check if this upgrade was already added
                    if expanded_content.split("\n")[0] in current_content:
                        return

                    # Append the expansion
                    new_content = f"{current_content}\n\n---\n\n{expanded_content}"

                    # Discord message limit check
                    if len(new_content) > 2000:
                        # Send as separate message if too long
                        await channel.send(expanded_content)
                    else:
                        await message.edit(content=new_content)

                    logger.info(
                        f"✅ Applied upgrade reaction {payload.emoji} to message {payload.message_id}"
                    )

                except discord.NotFound:
                    logger.warning(
                        f"⚠️ Message {payload.message_id} not found for upgrade"
                    )
                except discord.Forbidden:
                    logger.warning(
                        f"⚠️ No permission to edit message {payload.message_id}"
                    )
                except Exception as e:
                    logger.error(
                        f"❌ Error applying upgrade to message {payload.message_id}: {e}"
                    )

        except Exception as e:
            logger.error(f"❌ Error handling reaction upgrade: {e}")

    async def cog_unload(self):
        """Cleanup when cog is unloaded."""
        await self.upgrade_manager.cleanup_expired_cache()
        logger.info("🧹 ImageUpgradeCommands cog unloaded")


async def setup(bot) -> None:
    """Setup function for the cog."""
    await bot.add_cog(ImageUpgradeCommands(bot))
