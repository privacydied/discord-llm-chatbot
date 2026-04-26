# NVIDIA NIM Configuration Guide

## Quick Start

NVIDIA NIM reuses the existing OpenAI API infrastructure. To use NVIDIA NIM:

```bash
# In your .env file:
TEXT_BACKEND=nvidia
OPENAI_API_KEY=your_nvidia_api_key_here
OPENAI_API_BASE=https://integrate.api.nvidia.com/v1
OPENAI_TEXT_MODEL=meta/llama3-70b-instruct
```

That's it! The bot will automatically use NVIDIA NIM endpoints when `TEXT_BACKEND=nvidia`.

## How It Works

When `TEXT_BACKEND=nvidia` is set, the bot:
1. Uses `OPENAI_API_KEY` as the NVIDIA API key
2. Uses `OPENAI_API_BASE` as the NVIDIA endpoint
3. Uses `OPENAI_TEXT_MODEL` for model selection
4. Routes all text generation through the existing OpenAI backend

This approach:
- ✅ Reuses existing, battle-tested OpenAI backend code
- ✅ Maintains compatibility with all OpenAI features
- ✅ Simplifies configuration (no new env vars to learn)
- ✅ Leverages existing retry/fallback logic
- ✅ Works with streaming, context, and all other features

## Environment Variables

### Required (when TEXT_BACKEND=nvidia)

| Variable | Description | Example |
|----------|-------------|---------|
| `TEXT_BACKEND` | Set to `nvidia` | `nvidia` |
| `OPENAI_API_KEY` | Your NVIDIA API key | `nvapi-...` |
| `OPENAI_API_BASE` | NVIDIA NIM endpoint | `https://integrate.api.nvidia.com/v1` |
| `OPENAI_TEXT_MODEL` | Model to use | `meta/llama3-70b-instruct` |

### Optional (for advanced configuration)

| Variable | Description | Default |
|----------|-------------|---------|
| `NVIDIA_NIM_API_KEY` | Alternative: NVIDIA API key | - |
| `NVIDIA_NIM_API_BASE` | Alternative: NVIDIA endpoint | `https://integrate.api.nvidia.com/v1` |
| `NVIDIA_NIM_TEXT_MODEL` | Alternative: Model name | `meta/llama3-70b-instruct` |
| `NVIDIA_NIM_FALLBACK_MODELS` | Fallback models | - |

Note: If both `OPENAI_*` and `NVIDIA_NIM_*` variables are set, the NVIDIA-specific ones take precedence when `TEXT_BACKEND=nvidia`.

## Supported Models

NVIDIA NIM provides access to optimized LLMs:

**Text Models:**
- `meta/llama3-70b-instruct` - Meta Llama 3 70B (recommended)
- `meta/llama3-8b-instruct` - Meta Llama 3 8B (faster)
- `mistralai/mistral-large` - Mistral Large
- `mistralai/mixtral-8x7b-instruct` - Mixtral 8x7B
- `google/gemma-7b` - Google Gemma 7B
- `google/gemma-2b` - Google Gemma 2B

Check [NVIDIA NIM Catalog](https://catalog.ngc.nvidia.com/models) for the latest available models.

## Examples

### Example 1: Basic NVIDIA NIM Setup

```bash
# .env file
TEXT_BACKEND=nvidia
OPENAI_API_KEY=nvapi-your-key-here
OPENAI_API_BASE=https://integrate.api.nvidia.com/v1
OPENAI_TEXT_MODEL=meta/llama3-70b-instruct
```

### Example 2: NVIDIA NIM with Fallback

```bash
# .env file
TEXT_BACKEND=nvidia
OPENAI_API_KEY=nvapi-your-key-here
OPENAI_API_BASE=https://integrate.api.nvidia.com/v1
OPENAI_TEXT_MODEL=meta/llama3-70b-instruct

# Fallback to OpenRouter if NVIDIA fails
TEXT_FALLBACK_MODELS=openrouter|deepseek/deepseek-chat-v3-0324:free
```

### Example 3: Switching Between Backends

```bash
# Use OpenAI/OpenRouter (default)
TEXT_BACKEND=openai
OPENAI_API_KEY=sk-...
OPENAI_API_BASE=https://openrouter.ai/api/v1
OPENAI_TEXT_MODEL=gpt-4

# OR use NVIDIA NIM
TEXT_BACKEND=nvidia
OPENAI_API_KEY=nvapi-...
OPENAI_API_BASE=https://integrate.api.nvidia.com/v1
OPENAI_TEXT_MODEL=meta/llama3-70b-instruct
```

## Migration from OpenAI/OpenRouter

To migrate from OpenAI/OpenRouter to NVIDIA NIM:

1. Keep your existing OpenAI configuration
2. Change `TEXT_BACKEND` to `nvidia`
3. Replace `OPENAI_API_KEY` with your NVIDIA API key
4. Update `OPENAI_API_BASE` to NVIDIA endpoint
5. Update `OPENAI_TEXT_MODEL` to a NVIDIA model

```bash
# Before (OpenRouter)
TEXT_BACKEND=openai
OPENAI_API_KEY=sk-or-...
OPENAI_API_BASE=https://openrouter.ai/api/v1
OPENAI_TEXT_MODEL=deepseek/deepseek-chat-v3-0324:free

# After (NVIDIA NIM)
TEXT_BACKEND=nvidia
OPENAI_API_KEY=nvapi-...
OPENAI_API_BASE=https://integrate.api.nvidia.com/v1
OPENAI_TEXT_MODEL=meta/llama3-70b-instruct
```

## Troubleshooting

### Authentication Error
```
NVIDIA NIM authentication failed - check API key
```
**Solution:** Verify your NVIDIA API key is correct and has not expired.

### Model Not Found
```
Model not found: meta/llama3-70b-instruct
```
**Solution:** Check that the model name is correct and available in your region.

### Rate Limiting
```
Rate limit exceeded
```
**Solution:** Reduce request frequency or check NVIDIA dashboard for limits.

## Getting API Key

1. Visit [NVIDIA NGC](https://ngc.nvidia.com/)
2. Sign up or log in
3. Navigate to API Keys
4. Generate a new key
5. Copy and save securely

## Performance Tips

- **For speed**: Use `meta/llama3-8b-instruct`
- **For quality**: Use `meta/llama3-70b-instruct`
- **For balance**: Use `mistralai/mixtral-8x7b-instruct`

## Limitations

- **Vision-Language**: NVIDIA NIM currently focuses on text models. Vision tasks will fall back to OpenAI/OpenRouter.
- **Model Availability**: Some models may not be available in all regions.
- **API Compatibility**: NVIDIA NIM uses OpenAI-compatible API but may not support all OpenAI-specific features.

## References

- [NVIDIA NIM Documentation](https://docs.nvidia.com/nim/)
- [NVIDIA API Catalog](https://catalog.ngc.nvidia.com/)
- [API Reference](https://docs.api.nvidia.com/)
