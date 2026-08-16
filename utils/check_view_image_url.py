"""Live check: vision inference straight from an http(s) URL, no temp file. [REH]

view_image depends on generate_vl_response accepting a URL, because see_infer
gates on os.path.exists and would reject one. This proves the assumption.

Usage: uv run python utils/check_view_image_url.py
"""

import asyncio

from dotenv import load_dotenv

load_dotenv("/volume1/py/discord-llm-chatbot/.env")

from bot.ai_backend import generate_vl_response  # noqa: E402
from bot.see import see_infer  # noqa: E402

# Stable, public, small.
IMAGE_URL = "https://picsum.photos/id/237/320/240"  # deterministic photo of a dog


async def main() -> None:
    print("1. see_infer with a URL (expected to fail — it requires a local path)")
    try:
        action = await see_infer(IMAGE_URL, prompt="What is in this image?")
        content = getattr(action, "content", "")
        print(f"   -> {str(content)[:140]}")
    except Exception as exc:
        print(f"   -> raised {type(exc).__name__}: {exc}")

    print("\n2. generate_vl_response with the same URL (the path view_image uses)")
    try:
        result = await generate_vl_response(image_url=IMAGE_URL, user_prompt="What animal is this? One sentence.")
    except Exception as exc:
        print(f"   -> FAILED {type(exc).__name__}: {exc}")
        return
    text = (result or {}).get("text") if isinstance(result, dict) else None
    print(f"   -> model: {(result or {}).get('model')}")
    print(f"   -> text : {str(text)[:300]}")


if __name__ == "__main__":
    asyncio.run(main())
