"""
Test script for the vision-to-text pipeline.
"""

import asyncio
import logging
from pathlib import Path
from bot.see import see_infer
from bot.brain import brain_infer

# Set up logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_vision_pipeline(image_path: str, prompt: str = None):
    """Test the full vision-to-text pipeline."""
    try:
        # 1. Process image with VL model
        logger.info("👁️ Starting vision-language inference...")
        vl_result = await see_infer(Path(image_path), prompt or "What's in this image?")
        logger.info("✅ VL model output:")
        print("\nVL Model Output:")
        print("-" * 80)
        print(vl_result)
        print("-" * 80)

        # 2. Create enhanced prompt for text model
        enhanced_prompt = f"""Based on this image description:
        
{vl_result}

Answer this question: {prompt or "What can you tell me about this image?"}
"""

        # 3. Get text model response
        logger.info("🤖 Getting text model response...")
        text_response = await brain_infer(enhanced_prompt)

        # 4. Print final response
        print("\n📝 Final Response:")
        print("-" * 80)
        print(text_response)
        print("-" * 80)

    except Exception as e:
        logger.error(f"❌ Error in test_vision_pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python test_vision_pipeline.py <image_path> [prompt]")
        sys.exit(1)

    image_path = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else None

    asyncio.run(test_vision_pipeline(image_path, prompt))
