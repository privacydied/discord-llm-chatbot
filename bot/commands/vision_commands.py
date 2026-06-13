"""Vision Generation Slash Commands.

Implements Discord slash commands for image and video generation:
- /image: Text-to-image generation
- /imgedit: Image editing and variations
- /video: Text-to-video generation
- /vidref: Image-to-video animation

Follows existing bot patterns and integrates with Vision orchestration system.
"""

import asyncio
import tempfile
from datetime import UTC
from pathlib import Path
from typing import Literal

import aiohttp
import discord
from discord import app_commands
from discord.ext import commands

from bot.config import load_config
from bot.utils.logging import get_logger
from bot.vision.orchestrator import VisionOrchestrator
from bot.vision.types import VisionError, VisionProvider, VisionRequest, VisionTask

logger = get_logger(__name__)


class VisionCommands(commands.Cog):
    """Vision generation slash commands cog."""

    def __init__(self, bot) -> None:
        self.bot = bot
        self.config = load_config()
        self.logger = logger

        # Initialize orchestrator lazily
        self._orchestrator = None

        # Track temp files created by _download_attachment for cleanup
        self._temp_files: set[Path] = set()

        self.logger.info("Vision commands cog initialized")

    @property
    def orchestrator(self) -> VisionOrchestrator:
        """Lazy-load vision orchestrator."""
        if self._orchestrator is None:
            self._orchestrator = VisionOrchestrator(self.config)
        return self._orchestrator

    async def cog_unload(self) -> None:
        """Clean up temp files on cog unload."""
        for path in self._temp_files:
            try:
                if path.exists():
                    path.unlink()
            except OSError:
                pass
        self._temp_files.clear()

    @app_commands.command(name="image", description="Generate images from text prompts")
    @app_commands.describe(
        prompt="Text description of the image to generate",
        size="Image dimensions (default: square)",
        steps="Number of diffusion steps (10-50, default: 30)",
        guidance="Guidance scale for prompt adherence (1-20, default: 7)",
        negative="Things to avoid in the image",
        seed="Random seed for reproducible results",
        count="Number of images to generate (1-4, default: 1)",
        provider="Vision provider to use (default: auto)",
        model="Specific model to use (default: auto)",
    )
    async def image_command(
        self,
        interaction: discord.Interaction,
        prompt: str,
        size: Literal["square", "portrait", "landscape", "4k"] | None = "square",
        steps: app_commands.Range[int, 10, 50] | None = 30,
        guidance: app_commands.Range[float, 1.0, 20.0] | None = 7.0,
        negative: str | None = None,
        seed: int | None = None,
        count: app_commands.Range[int, 1, 4] | None = 1,
        provider: Literal["together", "novita", "openrouter", "auto"] | None = "auto",
        model: str | None = None,
    ) -> None:
        """Handle /image slash command."""
        # Check if Vision is enabled
        if not self.config.get("VISION_ENABLED", False):
            await interaction.response.send_message("🚫 Vision generation is currently disabled.", ephemeral=True)
            return

        # Map size parameter to dimensions
        size_mapping = {
            "square": (1024, 1024),
            "portrait": (768, 1024),
            "landscape": (1024, 768),
            "4k": (2048, 2048),
        }
        width, height = size_mapping[size]

        # Create vision request
        request = VisionRequest(
            task=VisionTask.TEXT_TO_IMAGE,
            prompt=prompt,
            user_id=str(interaction.user.id),
            guild_id=str(interaction.guild.id) if interaction.guild else None,
            channel_id=str(interaction.channel.id),
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance,
            negative_prompt=negative or "",
            seed=seed,
            batch_size=count,
            preferred_provider=VisionProvider(provider) if provider != "auto" else None,
            preferred_model=model,
        )

        # Set Discord interaction context
        request.discord_interaction_id = str(interaction.id)

        try:
            # Submit job to orchestrator
            job = await self.orchestrator.submit_job(request)

            # Send initial response with job info
            embed = discord.Embed(
                title="🎨 Image Generation Started",
                color=0x00FF00,
                description=f"**Prompt:** {prompt[:100]}{'...' if len(prompt) > 100 else ''}",
            )
            embed.add_field(name="Job ID", value=f"`{job.job_id[:8]}`", inline=True)
            embed.add_field(name="Size", value=f"{width}×{height}", inline=True)
            embed.add_field(name="Count", value=str(count), inline=True)
            embed.add_field(
                name="Estimated Cost",
                value=f"${job.request.estimated_cost:.3f}",
                inline=True,
            )
            embed.add_field(
                name="Provider",
                value=job.provider_assigned.value if job.provider_assigned else "Auto",
                inline=True,
            )
            embed.add_field(name="Status", value="🟡 Queued", inline=True)

            # Set footer with timestamp
            embed.set_footer(text="Generation in progress...")

            await interaction.response.send_message(
                embed=embed,
                ephemeral=self.config.get("VISION_EPHEMERAL_RESPONSES", True),
            )

            # Start background progress monitoring
            _task = asyncio.create_task(
                self._monitor_job_progress(interaction, job),
                name=f"vision-monitor-{job.job_id}",
            )
            _task.add_done_callback(lambda t: self.logger.debug("job monitor complete", extra={"detail": {"job_id": job.job_id}}) if t.done() and not t.cancelled() and t.exception() else None)

        except VisionError as e:
            self.logger.exception(
                "Vision generation failed",
                extra={
                    "event": "vision.image.error",
                    "detail": {
                        "error_type": e.error_type.value,
                        "message": e.message,
                        "user_message": e.user_message,
                    },
                },
            )

            embed = discord.Embed(
                title="❌ Image Generation Failed",
                description=e.user_message,
                color=0xFF0000,
            )

            await interaction.response.send_message(embed=embed, ephemeral=True)

        except Exception as e:
            self.logger.error(
                "Unexpected error in image command",
                exc_info=True,
                extra={
                    "event": "vision.image.exception",
                    "detail": {"error": str(e), "error_type": type(e).__name__},
                },
            )

            await interaction.response.send_message("❌ An unexpected error occurred. Please try again.", ephemeral=True)

    @app_commands.command(name="imgedit", description="Edit, modify, or create variations of images")
    @app_commands.describe(
        image="Image to edit or modify",
        prompt="Description of desired changes",
        strength="How much to change the image (0.1-1.0, default: 0.8)",
        steps="Number of diffusion steps (10-50, default: 30)",
        guidance="Guidance scale (1-20, default: 7)",
        negative="Things to avoid in the result",
        seed="Random seed for reproducible results",
        provider="Vision provider to use (default: auto)",
        model="Specific model to use (default: auto)",
    )
    async def imgedit_command(
        self,
        interaction: discord.Interaction,
        image: discord.Attachment,
        prompt: str,
        strength: app_commands.Range[float, 0.1, 1.0] | None = 0.8,
        steps: app_commands.Range[int, 10, 50] | None = 30,
        guidance: app_commands.Range[float, 1.0, 20.0] | None = 7.0,
        negative: str | None = None,
        seed: int | None = None,
        provider: Literal["together", "novita", "auto"] | None = "auto",
        model: str | None = None,
    ) -> None:
        """Handle /imgedit slash command."""
        if not self.config.get("VISION_ENABLED", False):
            await interaction.response.send_message("🚫 Vision generation is currently disabled.", ephemeral=True)
            return

        # Validate attachment
        if not image.filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            await interaction.response.send_message("❌ Please provide a valid image file (JPG, PNG, WebP).", ephemeral=True)
            return

        if image.size > 25 * 1024 * 1024:  # 25MB limit
            await interaction.response.send_message("❌ Image file too large. Maximum size: 25MB.", ephemeral=True)
            return

        try:
            # Download attachment to temporary file
            input_image_path = await self._download_attachment(image)

            # Create vision request
            request = VisionRequest(
                task=VisionTask.IMAGE_TO_IMAGE,
                prompt=prompt,
                user_id=str(interaction.user.id),
                guild_id=str(interaction.guild.id) if interaction.guild else None,
                channel_id=str(interaction.channel.id),
                input_image=input_image_path,
                strength=strength,
                steps=steps,
                guidance_scale=guidance,
                negative_prompt=negative or "",
                seed=seed,
                preferred_provider=VisionProvider(provider) if provider != "auto" else None,
                preferred_model=model,
            )

            request.discord_interaction_id = str(interaction.id)

            # Submit job
            job = await self.orchestrator.submit_job(request)

            # Send initial response
            embed = discord.Embed(
                title="✏️ Image Editing Started",
                color=0x00FF00,
                description=f"**Prompt:** {prompt[:100]}{'...' if len(prompt) > 100 else ''}",
            )
            embed.add_field(name="Job ID", value=f"`{job.job_id[:8]}`", inline=True)
            embed.add_field(name="Strength", value=f"{strength:.1f}", inline=True)
            embed.add_field(name="Steps", value=str(steps), inline=True)
            embed.add_field(
                name="Estimated Cost",
                value=f"${job.request.estimated_cost:.3f}",
                inline=True,
            )
            embed.add_field(
                name="Provider",
                value=job.provider_assigned.value if job.provider_assigned else "Auto",
                inline=True,
            )
            embed.add_field(name="Status", value="🟡 Queued", inline=True)

            # Show original image as thumbnail
            embed.set_thumbnail(url=image.url)
            embed.set_footer(text="Processing your image...")

            await interaction.response.send_message(
                embed=embed,
                ephemeral=self.config.get("VISION_EPHEMERAL_RESPONSES", True),
            )

            # Monitor progress
            _task = asyncio.create_task(
                self._monitor_job_progress(interaction, job),
                name=f"vision-monitor-imgedit-{job.job_id}",
            )
            job_id = job.job_id
            _task.add_done_callback(
                lambda t: (
                    self.logger.error(
                        f"imgedit monitor task failed: {t.exception()}",
                        exc_info=t.exception(),
                        extra={"detail": {"job_id": job_id}},
                    )
                    if t.done() and not t.cancelled() and t.exception()
                    else None
                ),
            )

        except Exception as e:
            self.logger.error(
                "Error in imgedit command",
                exc_info=True,
                extra={
                    "event": "vision.imgedit.exception",
                    "detail": {"error": str(e), "error_type": type(e).__name__},
                },
            )
            await interaction.response.send_message(
                "❌ Failed to process image editing request. Please try again.",
                ephemeral=True,
            )

    @app_commands.command(name="video", description="Generate videos from text prompts")
    @app_commands.describe(
        prompt="Text description of the video to generate",
        duration="Video length in seconds (1-10, default: 3)",
        fps="Frames per second (12-30, default: 24)",
        resolution="Video resolution (default: 720p)",
        style="Video style (default: natural)",
        seed="Random seed for reproducible results",
        provider="Vision provider to use (default: auto)",
        model="Specific model to use (default: auto)",
    )
    async def video_command(
        self,
        interaction: discord.Interaction,
        prompt: str,
        duration: app_commands.Range[int, 1, 10] | None = 3,
        fps: app_commands.Range[int, 12, 30] | None = 24,
        resolution: Literal["720p", "1080p"] | None = "720p",
        style: str | None = "natural",
        seed: int | None = None,
        provider: Literal["together", "novita", "auto"] | None = "auto",
        model: str | None = None,
    ) -> None:
        """Handle /video slash command."""
        if not self.config.get("VISION_ENABLED", False):
            await interaction.response.send_message("🚫 Vision generation is currently disabled.", ephemeral=True)
            return

        # Map resolution to dimensions
        res_mapping = {"720p": (1280, 720), "1080p": (1920, 1080)}
        width, height = res_mapping[resolution]

        try:
            # Create vision request
            request = VisionRequest(
                task=VisionTask.TEXT_TO_VIDEO,
                prompt=prompt,
                user_id=str(interaction.user.id),
                guild_id=str(interaction.guild.id) if interaction.guild else None,
                channel_id=str(interaction.channel.id),
                width=width,
                height=height,
                duration_seconds=duration,
                fps=fps,
                style=style,
                seed=seed,
                preferred_provider=VisionProvider(provider) if provider != "auto" else None,
                preferred_model=model,
            )

            request.discord_interaction_id = str(interaction.id)

            # Submit job
            job = await self.orchestrator.submit_job(request)

            # Send initial response
            embed = discord.Embed(
                title="🎬 Video Generation Started",
                color=0x00FF00,
                description=f"**Prompt:** {prompt[:100]}{'...' if len(prompt) > 100 else ''}",
            )
            embed.add_field(name="Job ID", value=f"`{job.job_id[:8]}`", inline=True)
            embed.add_field(name="Duration", value=f"{duration}s", inline=True)
            embed.add_field(name="Resolution", value=resolution, inline=True)
            embed.add_field(name="FPS", value=str(fps), inline=True)
            embed.add_field(
                name="Estimated Cost",
                value=f"${job.request.estimated_cost:.3f}",
                inline=True,
            )
            embed.add_field(name="Status", value="🟡 Queued", inline=True)

            embed.set_footer(text="Video generation may take several minutes...")

            await interaction.response.send_message(
                embed=embed,
                ephemeral=self.config.get("VISION_EPHEMERAL_RESPONSES", True),
            )

            # Monitor progress (video takes longer)
            _task = asyncio.create_task(
                self._monitor_job_progress(interaction, job, long_running=True),
                name=f"vision-monitor-video-{job.job_id}",
            )
            job_id = job.job_id
            _task.add_done_callback(
                lambda t: (
                    self.logger.error(
                        f"video monitor task failed: {t.exception()}",
                        exc_info=t.exception(),
                        extra={"detail": {"job_id": job_id}},
                    )
                    if t.done() and not t.cancelled() and t.exception()
                    else None
                ),
            )

        except Exception as e:
            self.logger.error(
                "Error in video command",
                exc_info=True,
                extra={
                    "event": "vision.video.exception",
                    "detail": {"error": str(e), "error_type": type(e).__name__},
                },
            )
            await interaction.response.send_message("❌ Failed to start video generation. Please try again.", ephemeral=True)

    @app_commands.command(name="vidref", description="Animate images into videos")
    @app_commands.describe(
        image="Image to animate into video",
        prompt="Optional description of desired motion/changes",
        duration="Video length in seconds (1-8, default: 3)",
        fps="Frames per second (12-30, default: 24)",
        mode="Animation mode (default: image2video)",
        seed="Random seed for reproducible results",
        provider="Vision provider to use (default: auto)",
        model="Specific model to use (default: auto)",
    )
    async def vidref_command(
        self,
        interaction: discord.Interaction,
        image: discord.Attachment,
        prompt: str | None = None,
        duration: app_commands.Range[int, 1, 8] | None = 3,
        fps: app_commands.Range[int, 12, 30] | None = 24,
        mode: Literal["image2video", "start_end"] | None = "image2video",
        seed: int | None = None,
        provider: Literal["together", "novita", "auto"] | None = "auto",
        model: str | None = None,
    ) -> None:
        """Handle /vidref slash command."""
        if not self.config.get("VISION_ENABLED", False):
            await interaction.response.send_message("🚫 Vision generation is currently disabled.", ephemeral=True)
            return

        # Validate attachment
        if not image.filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            await interaction.response.send_message("❌ Please provide a valid image file (JPG, PNG, WebP).", ephemeral=True)
            return

        if image.size > 25 * 1024 * 1024:
            await interaction.response.send_message("❌ Image file too large. Maximum size: 25MB.", ephemeral=True)
            return

        try:
            # Download attachment
            input_image_path = await self._download_attachment(image)

            # Create vision request
            request = VisionRequest(
                task=VisionTask.IMAGE_TO_VIDEO,
                prompt=prompt or "Animate this image naturally",
                user_id=str(interaction.user.id),
                guild_id=str(interaction.guild.id) if interaction.guild else None,
                channel_id=str(interaction.channel.id),
                input_image=input_image_path,
                duration_seconds=duration,
                fps=fps,
                mode=mode,
                seed=seed,
                preferred_provider=VisionProvider(provider) if provider != "auto" else None,
                preferred_model=model,
            )

            request.discord_interaction_id = str(interaction.id)

            # Submit job
            job = await self.orchestrator.submit_job(request)

            # Send initial response
            embed = discord.Embed(
                title="🎞️ Image-to-Video Started",
                color=0x00FF00,
                description=prompt or "Animating your image...",
            )
            embed.add_field(name="Job ID", value=f"`{job.job_id[:8]}`", inline=True)
            embed.add_field(name="Duration", value=f"{duration}s", inline=True)
            embed.add_field(name="FPS", value=str(fps), inline=True)
            embed.add_field(name="Mode", value=mode, inline=True)
            embed.add_field(
                name="Estimated Cost",
                value=f"${job.request.estimated_cost:.3f}",
                inline=True,
            )
            embed.add_field(name="Status", value="🟡 Queued", inline=True)

            # Show reference image
            embed.set_thumbnail(url=image.url)
            embed.set_footer(text="Video generation may take several minutes...")

            await interaction.response.send_message(
                embed=embed,
                ephemeral=self.config.get("VISION_EPHEMERAL_RESPONSES", True),
            )

            # Monitor progress
            _task = asyncio.create_task(
                self._monitor_job_progress(interaction, job, long_running=True),
                name=f"vision-monitor-vidref-{job.job_id}",
            )
            job_id = job.job_id
            _task.add_done_callback(
                lambda t: (
                    self.logger.error(
                        f"vidref monitor task failed: {t.exception()}",
                        exc_info=t.exception(),
                        extra={"detail": {"job_id": job_id}},
                    )
                    if t.done() and not t.cancelled() and t.exception()
                    else None
                ),
            )

        except Exception as e:
            self.logger.error(
                "Error in vidref command",
                exc_info=True,
                extra={
                    "event": "vision.vidref.exception",
                    "detail": {"error": str(e), "error_type": type(e).__name__},
                },
            )
            await interaction.response.send_message(
                "❌ Failed to start image-to-video generation. Please try again.",
                ephemeral=True,
            )

    async def _download_attachment(self, attachment: discord.Attachment) -> Path:
        """Download Discord attachment to temporary file [RM]."""
        # Sanitize suffix from the original filename, fall back to .bin
        suffix = Path(attachment.filename).suffix or ".bin"
        # Only allow known safe suffixes to prevent writing executable content
        safe_suffixes = {
            ".mp3",
            ".wav",
            ".ogg",
            ".flac",
            ".m4a",
            ".mp4",
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".webp",
            ".bin",
            ".txt",
            ".pdf",
            ".csv",
            ".json",
            ".xml",
            ".yml",
            ".yaml",
            ".md",
            ".log",
        }
        if suffix.lower() not in safe_suffixes:
            suffix = ".bin"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_path = Path(temp_file.name)
        temp_file.close()
        self._temp_files.add(temp_path)

        # Download attachment data
        async with aiohttp.ClientSession() as session, session.get(attachment.url) as resp:
            if resp.status == 200:
                data = await resp.read()
                with open(temp_path, "wb") as f:
                    f.write(data)
            else:
                msg = f"Failed to download attachment: HTTP {resp.status}"
                raise Exception(msg)

        return temp_path

    async def _monitor_job_progress(self, interaction: discord.Interaction, job, long_running: bool = False) -> None:
        """Monitor job progress and update Discord message [PA]."""
        update_interval = self.config.get("VISION_PROGRESS_UPDATE_INTERVAL_S", 10)
        max_updates = 60 if long_running else 30  # More updates for video

        try:
            for update_count in range(max_updates):
                await asyncio.sleep(update_interval)

                # Get current job status
                current_job = await self.orchestrator.get_job_status(job.job_id)
                if not current_job:
                    break

                # Check if job completed
                if current_job.is_terminal_state():
                    if current_job.state.value == "completed" and current_job.response:
                        await self._send_completion_message(interaction, current_job)
                    else:
                        await self._send_error_message(interaction, current_job)
                    break

                # Send periodic progress updates
                if update_count % 3 == 0:  # Every 3rd update (30s intervals)
                    await self._send_progress_update(interaction, current_job)

        except Exception as e:
            self.logger.error(
                "Error monitoring job progress",
                exc_info=True,
                extra={
                    "event": "vision.progress.exception",
                    "detail": {"error": str(e), "error_type": type(e).__name__},
                },
            )

    async def _send_progress_update(self, interaction: discord.Interaction, job) -> None:
        """Send progress update embed [UX]."""
        try:
            embed = discord.Embed(
                title=f"{'🎨' if job.request.task == VisionTask.TEXT_TO_IMAGE else '🎬'} Generation in Progress",
                color=0xFFAA00,
            )
            embed.add_field(name="Job ID", value=f"`{job.job_id[:8]}`", inline=True)
            embed.add_field(name="Status", value=f"🟡 {job.state.value.title()}", inline=True)
            embed.add_field(name="Progress", value=f"{job.progress_percentage}%", inline=True)

            # Add provider info if available
            if job.provider_assigned:
                embed.add_field(name="Provider", value=job.provider_assigned.value, inline=True)

            embed.set_footer(text=f"Elapsed: {self._format_elapsed_time(job)}")

            # Try to edit the original response
            await interaction.edit_original_response(embed=embed)

        except discord.NotFound:
            # Original message was deleted, stop monitoring
            pass
        except Exception as e:
            self.logger.debug(f"Could not update progress: {e}")

    async def _send_completion_message(self, interaction: discord.Interaction, job) -> None:
        """Send completion message with generated artifacts [UX]."""
        try:
            response = job.response
            task_icons = {
                VisionTask.TEXT_TO_IMAGE: "🎨",
                VisionTask.IMAGE_TO_IMAGE: "✏️",
                VisionTask.TEXT_TO_VIDEO: "🎬",
                VisionTask.IMAGE_TO_VIDEO: "🎞️",
            }

            embed = discord.Embed(
                title=f"{task_icons.get(job.request.task, '✅')} Generation Complete!",
                color=0x00FF00,
                description=f"**Prompt:** {job.request.prompt[:150]}{'...' if len(job.request.prompt) > 150 else ''}",
            )

            embed.add_field(name="Job ID", value=f"`{job.job_id[:8]}`", inline=True)
            embed.add_field(name="Provider", value=response.model_used, inline=True)
            embed.add_field(
                name="Processing Time",
                value=f"{response.processing_time_seconds:.1f}s",
                inline=True,
            )
            embed.add_field(name="Actual Cost", value=f"${response.actual_cost:.3f}", inline=True)
            embed.add_field(
                name="File Size",
                value=f"{response.file_size_bytes / (1024 * 1024):.1f}MB",
                inline=True,
            )

            # Add dimensions for images or duration for videos
            if response.dimensions:
                embed.add_field(
                    name="Dimensions",
                    value=f"{response.dimensions[0]}×{response.dimensions[1]}",
                    inline=True,
                )
            elif response.duration_seconds:
                embed.add_field(
                    name="Duration",
                    value=f"{response.duration_seconds:.1f}s",
                    inline=True,
                )

            # Prepare file attachments
            files = []
            for artifact_path in response.artifacts:
                if artifact_path.exists() and artifact_path.stat().st_size < 100 * 1024 * 1024:  # 100MB Discord limit
                    discord_file = discord.File(artifact_path, filename=artifact_path.name)
                    files.append(discord_file)

            if files:
                # Send final message with files
                await interaction.followup.send(
                    embed=embed,
                    files=files,
                    ephemeral=False,  # Make final result visible to all
                )

                # Update original message to show completion
                completion_embed = discord.Embed(
                    title="✅ Generation Complete",
                    description="Your generated content has been posted above.",
                    color=0x00FF00,
                )
                await interaction.edit_original_response(embed=completion_embed)
            else:
                embed.add_field(
                    name="⚠️ Note",
                    value="Files too large for Discord upload",
                    inline=False,
                )
                await interaction.edit_original_response(embed=embed)

        except Exception as e:
            self.logger.error(
                "Error sending completion message",
                exc_info=True,
                extra={
                    "event": "vision.completion.exception",
                    "detail": {"error": str(e), "error_type": type(e).__name__},
                },
            )

    async def _send_error_message(self, interaction: discord.Interaction, job) -> None:
        """Send error message for failed jobs [REH]."""
        try:
            error = job.error or job.response.error if job.response else None

            embed = discord.Embed(title="❌ Generation Failed", color=0xFF0000)
            embed.add_field(name="Job ID", value=f"`{job.job_id[:8]}`", inline=True)
            embed.add_field(name="Status", value=job.state.value.title(), inline=True)

            if error:
                embed.description = error.user_message
                embed.add_field(name="Error Type", value=error.error_type.value, inline=True)
            else:
                embed.description = "Generation failed for an unknown reason."

            embed.set_footer(text="You can try again with different parameters.")

            await interaction.edit_original_response(embed=embed)

        except Exception as e:
            self.logger.error(
                "Error sending error message",
                exc_info=True,
                extra={
                    "event": "vision.error_msg.exception",
                    "detail": {"error": str(e), "error_type": type(e).__name__},
                },
            )

    def _format_elapsed_time(self, job) -> str:
        """Format elapsed time since job started [CMV]."""
        if not job.started_at:
            return "Not started"

        from datetime import datetime

        elapsed = datetime.now(UTC) - job.started_at
        total_seconds = int(elapsed.total_seconds())

        if total_seconds < 60:
            return f"{total_seconds}s"
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes}m {seconds}s"


async def setup(bot) -> None:
    """Setup function for Discord cog loading."""
    await bot.add_cog(VisionCommands(bot))
