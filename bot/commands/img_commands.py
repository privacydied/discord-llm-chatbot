"""
!img prefix command implementation

Provides traditional prefix command interface for image generation that works
with or without bot mention in guild channels. Delegates to the existing
vision orchestrator system for actual processing.
"""

import discord
from discord.ext import commands
from typing import Optional

from bot.util.logging import get_logger
from bot.config import load_config

logger = get_logger(__name__)


class ImgCommands(commands.Cog):
    """Traditional !img prefix command cog"""
    
    def __init__(self, bot):
        self.bot = bot
        self.config = load_config()
        self.logger = logger
        
        self.logger.info("!img prefix commands cog initialized")
    
    @commands.command(name="img", aliases=["image"], help="Generate images from text prompts")
    async def img_command(self, ctx, *, prompt: Optional[str] = None):
        """
        Handle !img prefix command - delegates to vision generation system
        
        Usage:
        !img a kitten playing with yarn
        @Bot !img a sunset over mountains
        """
        if not prompt:
            embed = discord.Embed(
                title="🎨 Image Generation Help",
                description="Generate images from text descriptions!",
                color=0x00ff00
            )
            embed.add_field(
                name="Usage", 
                value="!img <description>\n!img a kitten playing with yarn\n!img sunset over mountains",
                inline=False
            )
            embed.add_field(
                name="Note", 
                value="Works with or without mentioning me in guild channels",
                inline=False
            )
            await ctx.send(embed=embed)
            return
        
        # Log command detection
        self.logger.info(f"Found command 'IMG', delegating to vision system (msg_id: {ctx.message.id})")
        
        # Check if Vision is enabled
        if not self.config.get("VISION_ENABLED", False):
            await ctx.send("🚫 Vision generation is currently disabled.", ephemeral=True)
            return
        
        # Delegate to router's vision generation handler
        # Import here to avoid circular imports
        from bot.vision.intent_router import VisionIntentResult, VisionIntentParams
        from bot.vision.types import VisionTask
        
        # Create a mock intent result that matches what the vision system expects
        class MockIntentParams:
            def __init__(self, prompt: str):
                self.task = VisionTask.TEXT_TO_IMAGE.value
                self.prompt = prompt
                self.negative_prompt = ""
                self.width = 1024
                self.height = 1024
                self.steps = 30
                self.guidance_scale = 7.0
                self.seed = None
                self.preferred_provider = None
        
        class MockIntentResult:
            def __init__(self, prompt: str):
                self.extracted_params = MockIntentParams(prompt)
        
        # Get router from bot and delegate
        if hasattr(self.bot, 'router') and self.bot.router:
            try:
                mock_intent = MockIntentResult(prompt)
                action = await self.bot.router._handle_vision_generation(
                    mock_intent, 
                    ctx.message, 
                    ""  # context_str
                )
                
                # The vision handler manages its own response, so we don't need to do anything more
                self.logger.info(f"Successfully delegated !img to vision system (msg_id: {ctx.message.id})")
                
            except Exception as e:
                self.logger.error(f"Failed to delegate !img to vision system: {e}", exc_info=True)
                await ctx.send("❌ Failed to process image generation request. Please try again.")
        else:
            await ctx.send("🚫 Vision system is not available right now. Please try again later.")


async def setup(bot):
    """Setup function for Discord cog loading"""
    await bot.add_cog(ImgCommands(bot))
