"""
Hybrid multimodal pipeline controller
"""
import logging
from pathlib import Path
from .exceptions import InferenceError
from .brain import brain_infer
from .speak import speak_infer
from .see import see_infer
from .hear import hear_infer

logger = logging.getLogger(__name__)

async def hybrid_pipeline(ctx, content: str, mode: str = "both"):
    """Orchestrate multimodal processing pipeline"""
    try:
        logger.info(f"🚀 Starting hybrid pipeline in {mode} mode")
        
        # STT mode requires audio processing first
        if mode == "stt":
            if not ctx.message.attachments:
                await ctx.send("❌ Please provide an audio file for STT processing")
                return
            
            audio_path = await download_attachment(ctx.message.attachments[0])
            content = await hear_infer(audio_path)
            logger.info(f"👂 Transcribed audio: {content}")
            
            # After STT, continue with text processing
            mode = "text"

        # Vision-Language processing
        if mode == "vl":
            if not ctx.message.attachments:
                await ctx.send("❌ Please provide an image for vision processing")
                return
            
            image_path = await download_attachment(ctx.message.attachments[0])
            result = await see_infer(image_path, content)
            await ctx.send(result)
            return

        # Text processing core
        text_out = await brain_infer(content)
        replies = []

        # Text output modes
        if mode in ("text", "both"):
            replies.append(await ctx.send(text_out))

        # TTS output modes
        if mode in ("tts", "both"):
            try:
                audio_path = await speak_infer(text_out)
                replies.append(await ctx.send(file=discord.File(str(audio_path))))
            except InferenceError as e:
                await ctx.send(f"⚠️ TTS failed: {str(e)}")

        return replies
    except Exception as e:
        logger.error(f"🚨 Pipeline error: {str(e)}")
        await ctx.send("⚠️ An error occurred while processing your request")

async def download_attachment(attachment) -> Path:
    """Download Discord attachment to temporary file"""
    temp_file = Path(f"/tmp/{attachment.filename}")
    await attachment.save(temp_file)
    return temp_file