# NVIDIA NIM Integration Guide

## Overview

This bot now supports **NVIDIA NIM (NVIDIA Inference Microservices)** as a text backend option alongside OpenAI/OpenRouter and Ollama. NVIDIA NIM provides access to optimized LLMs including Meta Llama 3, Mistral AI models, Google Gemma, and more through an OpenAI-compatible API.

## Configuration

### Basic Setup

1. **Get NVIDIA API Key**: Obtain your API key from [NVIDIA NGC](https://ngc.nvidia.com/)

2. **Update `.env` file**:
```bash
# Enable NVIDIA NIM backend
TEXT_BACKEND=nvidia

# NVIDIA NIM credentials
NVIDIA_NIM_ENABLED=true
NVIDIA_NIM_API_KEY=your_nvidia_api_key_here
NVIDIA_NIM_API_BASE=https://integrate.api.nvidia.com/v1
NVIDIA_NIM_TEXT_MODEL=meta/llama3-70b-instruct

# Optional: Add NVIDIA NIM to fallback ladder
NVIDIA_NIM_FALLBACK_MODELS=meta/llama3-70b-instruct:30.0,mistralai/mistral-large:25.0
NVIDIA_NIM_PRIORITY=high
```

### Available NVIDIA NIM Models

Popular models available on NVIDIA NIM:

**Text Models:**
- `meta/llama3-70b-instruct` - Meta Llama 3 70B Instruct
- `meta/llama3-8b-instruct` - Meta Llama 3 8B Instruct  
- `mistralai/mistral-large` - Mistral Large
- `mistralai/mixtral-8x7b-instruct` - Mixtral 8x7B Instruct
- `google/gemma-7b` - Google Gemma 7B
- `google/gemma-2b` - Google Gemma 2B

**Note:** NVIDIA NIM currently focuses on text models. For vision-language tasks, the bot will automatically fall back to OpenAI/OpenRouter.

## Usage Modes

### Mode 1: Primary Backend

Use NVIDIA NIM as the default text generation backend:

```bash
TEXT_BACKEND=nvidia
NVIDIA_NIM_ENABLED=true
```

All text generation will use NVIDIA NIM endpoints.

### Mode 2: Fallback Ladder

Add NVIDIA NIM models to the fallback ladder alongside OpenRouter models:

```bash
# In .env.example format:
TEXT_FALLBACK_MODELS=nvidia|meta/llama3-70b-instruct,openrouter|deepseek/deepseek-chat-v3-0324:free
TEXT_FALLBACK_TIMEOUTS=30.0,25.0
```

Format: `provider|model_name:timeout_seconds`

### Mode 3: Hybrid Approach

Use NVIDIA NIM for primary text generation with OpenRouter as fallback:

```bash
TEXT_BACKEND=nvidia
NVIDIA_NIM_TEXT_MODEL=meta/llama3-70b-instruct

# Fallback to OpenRouter if NVIDIA fails
TEXT_FALLBACK_MODELS=openrouter|deepseek/deepseek-chat-v3-0324:free
```

## Advanced Configuration

### Timeout Settings

```bash
# Custom timeout for NVIDIA NIM requests
TEXTGEN_TIMEOUT_SECONDS=45

# Or use specific NVIDIA timeout
NVIDIA_NIM_TIMEOUT=30.0
```

### Priority Levels

Control where NVIDIA NIM appears in the fallback ladder:

```bash
# High: Try NVIDIA NIM first
NVIDIA_NIM_PRIORITY=high

# Medium: Try in the middle of the ladder
NVIDIA_NIM_PRIORITY=medium

# Low: Use as last resort
NVIDIA_NIM_PRIORITY=low
```

### Custom Base URL

For self-hosted NVIDIA NIM deployments:

```bash
NVIDIA_NIM_API_BASE=https://your-nim-server.com/v1
```

## Comparison: NVIDIA NIM vs OpenAI/OpenRouter

| Feature | NVIDIA NIM | OpenAI/OpenRouter |
|---------|-----------|-------------------|
| **Provider** | NVIDIA | OpenAI, Anthropic, Meta, etc. |
| **Base URL** | `integrate.api.nvidia.com` | `api.openai.com` or `openrouter.ai` |
| **Models** | Curated LLMs (Llama, Mistral, Gemma) | Wide variety (100+ models) |
| **Vision** | Limited/None | Extensive support |
| **Pricing** | Pay-per-token (NVIDIA pricing) | Varies by model/provider |
| **Latency** | Optimized inference | Varies by model |

## Troubleshooting

### Authentication Errors

```
NVIDIA NIM authentication failed - check NVIDIA_NIM_API_KEY
```

**Solution:** Verify your NVIDIA API key is correct and has not expired.

### Model Not Found

```
NVIDIA NIM HTTP error (404): Model not found
```

**Solution:** Check that the model name is correct and available in your region.

### Rate Limiting

```
NVIDIA NIM rate limit exceeded
```

**Solution:** Reduce request frequency or increase timeout settings.

### Timeout Errors

```
Request timeout: ...
```

**Solution:** Increase `TEXTGEN_TIMEOUT_SECONDS` or `NVIDIA_NIM_TIMEOUT`.

## Examples

### Example 1: Basic NVIDIA NIM Setup

```bash
# .env file
TEXT_BACKEND=nvidia
NVIDIA_NIM_ENABLED=true
NVIDIA_NIM_API_KEY=nim_abc123xyz
NVIDIA_NIM_TEXT_MODEL=meta/llama3-70b-instruct
```

### Example 2: NVIDIA NIM with OpenRouter Fallback

```bash
# .env file
TEXT_BACKEND=nvidia
NVIDIA_NIM_ENABLED=true
NVIDIA_NIM_API_KEY=nim_abc123xyz
NVIDIA_NIM_TEXT_MODEL=meta/llama3-70b-instruct

# Fallback configuration
TEXT_FALLBACK_MODELS=openrouter|deepseek/deepseek-chat-v3-0324:free
TEXT_FALLBACK_TIMEOUTS=25.0
```

### Example 3: Multiple NVIDIA Models in Ladder

```bash
# .env file
TEXT_BACKEND=openai  # Can still use openai as base

# Add NVIDIA NIM models to fallback ladder
TEXT_FALLBACK_MODELS=nvidia|meta/llama3-70b-instruct:30.0,nvidia|mistralai/mistral-large:25.0,openrouter|deepseek/deepseek-chat-v3-0324:free
```

## API Reference

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NVIDIA_NIM_ENABLED` | Enable NVIDIA NIM backend | `false` |
| `NVIDIA_NIM_API_KEY` | NVIDIA API key | - |
| `NVIDIA_NIM_API_BASE` | NVIDIA NIM endpoint | `https://integrate.api.nvidia.com/v1` |
| `NVIDIA_NIM_TEXT_MODEL` | Default text model | `meta/llama3-70b-instruct` |
| `NVIDIA_NIM_FALLBACK_MODELS` | Comma-separated fallback models | - |
| `NVIDIA_NIM_PRIORITY` | Priority in ladder | `high` |

### Supported Models

Check [NVIDIA NIM Catalog](https://catalog.ngc.nvidia.com/models) for the latest available models.

## Migration from OpenAI/OpenRouter

To migrate from OpenAI/OpenRouter to NVIDIA NIM:

1. Update `TEXT_BACKEND` to `nvidia`
2. Set `NVIDIA_NIM_API_KEY` to your NVIDIA API key
3. Choose appropriate `NVIDIA_NIM_TEXT_MODEL`
4. Test with a simple prompt
5. Adjust timeout settings if needed

```bash
# Before (OpenAI/OpenRouter)
TEXT_BACKEND=openai
OPENAI_API_KEY=sk-...
OPENAI_TEXT_MODEL=gpt-4

# After (NVIDIA NIM)
TEXT_BACKEND=nvidia
NVIDIA_NIM_API_KEY=nim_...
NVIDIA_NIM_TEXT_MODEL=meta/llama3-70b-instruct
```

## Performance Optimization

### Batch Processing
NVIDIA NIM performs well with batched requests. Consider batching if processing multiple prompts.

### Model Selection
- **Speed**: `meta/llama3-8b-instruct` (faster, smaller)
- **Quality**: `meta/llama3-70b-instruct` (slower, better quality)
- **Balance**: `mistralai/mixtral-8x7b-instruct` (good balance)

### Caching
Enable response caching to reduce API calls for repeated prompts.

## Security Notes

- Never commit your `NVIDIA_NIM_API_KEY` to version control
- Use environment variables or secure secret management
- Regularly rotate API keys
- Monitor usage and set appropriate limits

## Support

- NVIDIA NIM Documentation: https://docs.nvidia.com/nim/
- API Reference: https://docs.api.nvidia.com/
- Bot Issues: Check bot logs for detailed error messages
