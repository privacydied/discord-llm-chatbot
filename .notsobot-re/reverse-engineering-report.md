# NotSoBot API — Reverse Engineering Report

Source: official `NotSoBot/notsobot.ts` GitHub repo + live HTTP probes.
Date: 2026-08-27

## 1. Infrastructure

- Base URL: `https://notsobot.com`
- API path prefix: `/api`
- CDN: `https://cdn.notsobot.com`
- Beta: `https://beta.notsobot.com`
- Behind Cloudflare (cf-ray, cf-connecting-ip headers)
- Internal API server: `_nsb-api-01`
- Auth: `Authorization: Bot <token>` header (from `NOTSOBOT_API_TOKEN` env var)
- Token format: JWT (HS512), payload `{"id": "<user_id>", "exp": <unix_timestamp>}`

## 2. Authentication Headers

The bot sends these custom headers on every request:

| Header | Key | Value |
|--------|-----|-------|
| Authorization | `authorization` | `Bot <token>` |
| User ID | `x-user-id` | Discord user ID (base64 JSON in some contexts) |
| User | `x-user` | base64(JSON{avatar,discriminator,bot,id,username}) |
| Channel ID | `x-channel-id` | Discord channel ID |
| Guild ID | `x-guild-id` | Discord guild ID |
| Server ID | `x-server-id` | guildId or channelId |
| Server | `x-server` | base64(JSON channel object) for DMs |
| Server Owner | `x-server-owner-id` | guild owner ID |
| Entitlements | `x-entitlements` | base64(JSON Discord entitlements) |

## 3. ML Image Generation Endpoints

### 3a. `POST /api/utilities/ml/edit` — Image-to-Image / Edit

**Confirmed by live probe:** 405 for GET, 401 without auth.

Request body (JSON):
```
{
  "do_not_error": false,        // optional, default false
  "max_file_size": <number>,    // optional, auto-computed from context
  "model": "<string>",          // optional, e.g. "FLUX_KLEIN", "SDXL_TURBO"
  "query": "<string>",          // REQUIRED — text prompt
  "safe": true,                 // optional, default true
  "seed": <number>,             // optional
  "steps": <number>,            // optional
  "strength": <number>,         // optional
  "upload": <boolean>,          // optional
  "upload_expires_in": <number>,// optional
  "urls": ["<image_url>"]       // optional, image URL(s) to edit
}
```

Also supports `file` / `files` (multipart upload) and `url` for single image.

### 3b. `GET /api/utilities/ml/imagine` — Text-to-Image

**Method is GET** (not POST!) — query string parameters:

```
{
  "do_not_error": <boolean>,
  "max_file_size": <number>,
  "model": "<string>",
  "query": "<string>",          // REQUIRED
  "safe": <boolean>,
  "seed": <number>,
  "steps": <number>,
  "upload": <boolean>,
  "upload_expires_in": <number>
}
```

### 3c. `GET /api/utilities/ml/imagine/video` — Image-to-Video

Same parameters as imagine, but `safe` is hardcoded `true`.

### 3d. `POST /api/utilities/ml/interrogate` — Image Interrogation (captioning)

```
{ "url": "<image_url>" }
```
Also supports file upload.

### 3e. `POST /api/utilities/ml/mashup` — Image Mashup (multiple images)

```
{
  "do_not_error": <boolean>,
  "model": "<string>",
  "max_file_size": <number>,
  "safe": <boolean>,
  "seed": <number>,
  "steps": <number>,
  "strength": <number>,
  "upload": <boolean>,
  "upload_expires_in": <number>,
  "urls": ["<url1>", "<url2>", ...]
}
```

### 3f. `GET /api/jobs/:jobId` — Poll Job Status

Returns a `JobResponse` object. This is the async polling endpoint.

## 4. Job Response Schema (Polling)

```typescript
interface JobResponse {
  id: string;                    // job ID (UUID)
  status: string;                // "queued" | "running" | "completed" | "failed"
  user_id: string;               // Discord user ID
  result: {
    error: string | null;        // error message if failed
    response: FileResponse | null;
  };
}

interface FileResponse {
  arguments: Record<string, any> | null;
  file: {
    filename: string;
    filename_base: string;
    filename_safe: string;
    filename_safe_base: string;
    metadata: {
      duration: number,
      extension: string,
      framecount: number,
      height: number,
      mimetype: string,
      size: number,
      width: number
    },
    // ... binary file data or URL
  };
  storage: {
    expires_at: string | null;
    filename: string;
    id: string;
    metadata: { ... };
    temporary: boolean;
    urls: { ... };
  } | null;
  took: number;                  // processing time in ms
}
```

## 5. Available ML Diffusion Models

```typescript
enum MLDiffusionModels {
  FLUX_KLEIN = 'FLUX_KLEIN',     // "Flux Klein (Realistic)"
  SDXL_TURBO = 'SDXL_TURBO',     // "SDXL Turbo (Funny)"
}
```

## 6. Error Response Format

All API errors return JSON:
```json
{"code": 0, "message": "<description>", "status": <HTTP_STATUS>}
```

Live-confirmed error responses:
- `{"code":0,"message":"Unauthorized","status":401}` — missing/invalid token
- `{"code":0,"message":"Method GET not allowed for URL /utilities/ml/edit","status":405}`
- `{"code":0,"message":"Requested URL /v1 not found","status":404}`

## 7. Server-Timing

Response header `x-took: <ms>` — processing time on the server (observed 4-13ms for auth rejection).

## 8. Gaps in Local Plugin vs. Real API

| Aspect | Local Plugin (assumed) | Real API |
|--------|------------------------|----------|
| Text-to-image endpoint | Not implemented | `GET /api/utilities/ml/imagine` |
| Image edit endpoint | `POST /api/utilities/ml/edit` | Correct |
| Video generation | `stable-video-diffusion-img2vid-xt` model | `GET /api/utilities/ml/imagine/video` |
| Model names | `flux`, `stable-video-diffusion-img2vid-xt` | `FLUX_KLEIN`, `SDXL_TURBO` |
| Auth format | `Bot {api_key}` | Correct |
| Job poll | `GET /api/jobs/:id` | Correct |
| Interrogate | Not implemented | `POST /api/utilities/ml/interrogate` |
| Mashup | Not implemented | `POST /api/utilities/ml/mashup` |
| Edit sends | JSON body with `urls` array | Correct (also supports `file`/`files` multipart) |
| Imagine sends | Not implemented | GET query params (not POST body!) |
| Job response | `result.urls` array of URLs | `result.response.file` + `result.response.storage.urls` |

## 9. Key Findings

1. **The imagine endpoint uses GET, not POST** — the local plugin only implements edit (POST). Adding imagine support requires a GET request with query params.

2. **Model names are different** — the plugin uses `flux` but the real API uses `FLUX_KLEIN` and `SDXL_TURBO` as enum values.

3. **Job response structure is richer** — the real API returns `result.response.file` with metadata, not just a `urls` array. The plugin's `fetch_result` looks for `response_data.urls` which may need to map to `response.storage.urls` or `response.file`.

4. **The `/api` root is a debug echo** — returns request headers, client IP, and a JWT token. Useful for debugging but not a real API endpoint.

5. **Rate limiting exists** — rapid requests trigger HTTP 429.

6. **No public API documentation** — the only authoritative source is the GitHub source code. No OpenAPI/Swagger spec exists.

7. **The token from `/api` debug endpoint doesn't work** — it's a truncated/placeholder JWT (id=125), returns 401 on actual API calls.
