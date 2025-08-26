# Image Generation Guide

## Overview

The Discord LLM Chatbot supports AI image generation through Novita.ai's Qwen Image API. Users can generate images using natural language triggers in Discord.

## Configuration Status ✅

Your bot is **ready for image generation** with these settings in `yoroi.env`:

```bash
# Vision Generation System
VISION_ENABLED=true
VISION_API_KEY=sk_fVL1mAIms-9qH2VidKA_rr0L9vjSIUsmr0hpidYAgSE
VISION_ALLOWED_PROVIDERS=together,novita
VISION_DEFAULT_PROVIDER=novita
VISION_MODEL=qwen-image-txt2img
```

## How to Generate Images

### **Natural Language Triggers**

Users can trigger image generation with these phrases:
- `create an image`
- `generate an image` 
- `draw`
- `make a picture`
- `make a photo` ✨
- `create a photo` ✨
- `create a picture` ✨
- `paint`
- `sketch`
- `illustration`
- `artwork`
- `render`

### **Example Usage**

```
User: "create an image of a cyberpunk cityscape at night"
Bot: 🎨 Generating image... [progress updates]
     📸 Generated image successfully! [uploads image to Discord]
```

## Technical Implementation

### **API Flow**

1. **Request Submission**
   ```bash
   POST https://api.novita.ai/v3/async/qwen-image-txt2img
   Authorization: Bearer {API_KEY}
   Content-Type: application/json
   
   {
     "prompt": "A cyberpunk cityscape at night",
     "size": "1024*1024"
   }
   ```

2. **Task ID Response**
   ```json
   {"task_id": "uuid-task-id"}
   ```

3. **Polling for Results**
   ```bash
   GET https://api.novita.ai/v3/async/task-result?task_id={task_id}
   ```

4. **Final Result**
   ```json
   {
     "task": {"status": "TASK_STATUS_SUCCEED"},
     "images": [{
       "image_url": "https://cdn.novita.ai/output/image.jpeg",
       "image_type": "jpeg"
     }]
   }
   ```

### **Size Options**

Supported image sizes via `vision_policy.json`:
- **Portrait**: 768×1024
- **Landscape**: 1024×768  
- **Square**: 1024×1024
- **4K**: 2048×2048

Default: 1024×1024

### **Provider Architecture**

**Fixed Novita.ai Adapter** (`/bot/vision/providers/novita_adapter.py`):
- ✅ Correct endpoint: `/v3/async/qwen-image-txt2img`
- ✅ Simple payload format: `{"prompt": str, "size": str}`
- ✅ Proper polling with query parameter: `?task_id={id}`
- ✅ Async job handling with progress updates
- ✅ Error mapping and user-friendly messages

## File Structure

```
vision_data/
├── artifacts/          # Generated images
│   └── {job_id}_1.png
├── jobs/              # Job state tracking  
└── ledger.jsonl       # Cost tracking
```

## Cost & Limits

**Per Image**:
- Estimated cost: ~$0.03 USD
- Processing time: 10-30 seconds
- Max file size: 25MB
- TTL: 3 days

**Rate Limits**:
- Max concurrent jobs: 3 global
- Max per user: 1 concurrent
- Timeout: 5 minutes per job

## Error Handling

**Content Safety**: Prompts are filtered for inappropriate content
**Rate Limiting**: Automatic retry with backoff for 429 errors  
**Network Issues**: Graceful degradation with user-friendly messages
**Provider Outages**: Clear communication of service issues

## Monitoring

**Audit Logging**: All generation requests logged to `vision_data/ledger.jsonl`
**Metrics**: Available via Prometheus on port 8001
**Health Checks**: Provider availability monitoring

## Troubleshooting

### **Common Issues**

1. **"Vision generation is not properly configured"**
   - Check `VISION_API_KEY` in environment
   - Verify `VISION_ENABLED=true`

2. **"Content filtered by safety systems"**
   - Modify prompt to avoid potentially harmful content
   - Try more general, artistic descriptions

3. **"Rate limit exceeded"**
   - Wait 60 seconds before retrying
   - Reduce concurrent usage

4. **"Generation taking too long"**
   - Complex prompts may take 2-5 minutes
   - Check Novita.ai service status

### **Debug Commands**

```bash
# Check vision system status
python -m bot.commands.admin_commands vision_status

# Test API connectivity  
python tests/test_novita_image_generation.py

# View recent jobs
tail -f vision_data/ledger.jsonl
```

## Security & Privacy

- API keys stored securely in environment variables
- Generated images auto-expire after 3 days
- No prompt content stored long-term
- NSFW detection enabled by default
- Guild-level safety overrides supported

---

**Status**: ✅ **Ready for Production**  
**Last Updated**: 2025-08-25  
**API Version**: Novita.ai v3 (Qwen Image)
