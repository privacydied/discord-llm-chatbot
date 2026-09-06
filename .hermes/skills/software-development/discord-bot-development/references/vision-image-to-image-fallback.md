# Vision IMAGE_TO_IMAGE fallback via OpenRouter free VL models

When the primary image-edit provider (Novita) runs out of credit, the bot needs
a free fallback so `@Bot edit: <prompt>` and `/imgedit` still work. The pattern:
extend `OpenRouterPlugin` to support `IMAGE_TO_IMAGE` using a free VL model
(Qwen VL) via the chat completions API with image input.

## Problem

The vision provider ladder for `IMAGE_TO_IMAGE` was:
1. Novita (paid, requires balance) → fails with `QUOTA_EXCEEDED` when empty
2. Together (paid, often no credentials configured)
3. **No free fallback** → user gets "image generation unavailable"

`OpenRouterPlugin` only supported `TEXT_TO_IMAGE`, and wasn't in the default
`provider_order` for vision tasks.

## Solution (3 changes in `bot/vision/unified_adapter.py`)

### 1. Add IMAGE_TO_IMAGE to OpenRouterPlugin

```python
class OpenRouterPlugin(ProviderPlugin):
    def __init__(self, name, config, api_key):
        self.model_map = {
            VisionTask.TEXT_TO_IMAGE: "black-forest-labs/flux.2-pro",
            VisionTask.IMAGE_TO_IMAGE: "qwen/qwen2.5-vl-72b-instruct:free",
        }

    def capabilities(self):
        return {
            "modes": [VisionTask.TEXT_TO_IMAGE, VisionTask.IMAGE_TO_IMAGE],
            ...
        }
```

### 2. Build multi-part message for image input in submit()

When `task == IMAGE_TO_IMAGE` and `input_image_data` is present, send the image
as a base64 data URL in a multi-part content block alongside the prompt:

```python
if request.task == VisionTask.IMAGE_TO_IMAGE and request.input_image_data:
    b64_image = base64.b64encode(request.input_image_data).decode()
    # Detect content type from magic bytes
    ct = "image/png"
    if request.input_image_data[:3] == b"\xff\xd8\xff":
        ct = "image/jpeg"
    elif request.input_image_data[:4] == b"RIFF":
        ct = "image/webp"
    image_url = f"data:{ct};base64,{b64_image}"
    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": f"Edit this image: {request.prompt}"},
        ],
    }]
else:
    messages = [{"role": "user", "content": request.prompt}]
```

### 3. Add "openrouter" to default provider_order

```python
# In UnifiedVisionAdapter.submit(), the non-override path:
provider_order = policy.get(
    "provider_order",
    ["novita:qwen-image", "novita:txt2img", "together", "openrouter"],
)
```

OpenRouter is LAST so it only fires when paid providers fail.

## Why this works

- OpenRouter's chat completions API accepts image input for VL models
- `qwen/qwen2.5-vl-72b-instruct:free` is a free model that accepts images
- The `modalities: ["image", "text"]` response instruction tells OpenRouter
  the model should return an edited image
- The existing response-parsing logic in `OpenRouterPlugin.submit()` already
  extracts images from the response (methods 0-3 for various response shapes)

## Key pitfall: capabilities() must match

If you add `IMAGE_TO_IMAGE` to `capabilities()["modes"]` but the submit() method
doesn't actually handle it, the provider will be selected then fail with
`UNSUPPORTED_TASK`. Always implement both together.

## Testing

The existing test `test_edit_route_answers_honestly_when_no_provider_can_edit`
still works because it mocks `audit_task_coverage()` to return empty
`image_to_image` — the `_image_edit_unavailable_action()` gate fires before
the provider loop. In production, `audit_task_coverage()` reports OpenRouter
as capable, so the gate passes and the fallback works.

## Model selection

Free VL models on OpenRouter that support image input:
- `qwen/qwen2.5-vl-72b-instruct:free` (good quality, large context)
- `qwen/qwen2.5-vl-32b-instruct:free` (faster, smaller)
- `google/gemini-2.0-flash-exp:free` (Gemini, different strengths)

The `:free` suffix is mandatory — OpenRouter only guarantees zero-cost for
these variants. Paid models would appear in the ladder only if
`VISION_ALLOW_PAID_FALLBACK=1` is set.
