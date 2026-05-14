# Dashboard

Owner-only web dashboard for operational visibility and controlled bot actions.

## Overview

The dashboard provides:
- **Bot health and metrics** — uptime, guild count, latency, audit event count
- **Guild inventory** — connected servers with member counts, channel counts, and permission summaries
- **Audit log** — append-only, filterable, paginated event log for all dashboard actions
- **DM archive** — bot-visible DM conversations (users who DMed the bot or received DMs from the bot)
- **Send as bot** — owner-only DM and guild message sending with rate limiting and audit logging

## Configuration

All settings are controlled via environment variables in `.env`:

| Variable | Default | Description |
|---|---|---|
| `DASHBOARD_ENABLED` | `false` | Master toggle for the dashboard server |
| `DASHBOARD_HOST` | `127.0.0.1` | Bind address |
| `DASHBOARD_PORT` | `8011` | Bind port |
| `DASHBOARD_PUBLIC_BIND` | `false` | When true, binds to `DASHBOARD_HOST` instead of forcing localhost |
| `DASHBOARD_AUTH_TOKEN` | _(none)_ | **Required** when enabled. Strong random token for authentication |
| `DASHBOARD_SESSION_SECRET` | _(auto-generated)_ | Session signing key. Auto-generated if unset (printed once to console) |
| `DASHBOARD_OWNER_IDS` | _(falls back to `OWNER_IDS`)_ | Comma-separated Discord user IDs allowed to access the dashboard |
| `DASHBOARD_RATE_LIMIT_SENDS_PER_MINUTE` | `5` | Max DM/guild messages per minute per actor+target pair |
| `DASHBOARD_AUDIT_DB_PATH` | `./data/dashboard_audit.db` | SQLite path for audit log |
| `DASHBOARD_DM_ARCHIVE_ENABLED` | `true` | Enable DM archiving for bot-visible conversations |
| `DASHBOARD_DM_RETENTION_DAYS` | `90` | Soft-delete DM records older than this |
| `DASHBOARD_AUDIT_RETENTION_DAYS` | `180` | Delete audit records older than this |
| `DASHBOARD_MAX_MESSAGE_CHARS` | `1800` | Max content length for dashboard sends |
| `DASHBOARD_SHOW_MESSAGE_PREVIEWS` | `true` | Show message previews in audit log and DM viewer |

## Access

### Default (localhost only)

```
http://127.0.0.1:8011/
```

### SSH tunnel

```bash
ssh -L 8011:127.0.0.1:8011 user@server
# Then open http://127.0.0.1:8011/ locally
```

### Tailscale

```bash
DASHBOARD_ENABLED=true DASHBOARD_HOST=100.x.y.z DASHBOARD_PORT=8011
```

### Reverse proxy (recommended for external access)

```
DASHBOARD_ENABLED=true
DASHBOARD_PUBLIC_BIND=true
DASHBOARD_HOST=127.0.0.1  # Still bind to localhost; let proxy handle TLS
```

Then configure nginx/Caddy with HTTPS + basic auth/OIDC in front of `http://127.0.0.1:8011`.

## Security

- **Owner-only**: All API endpoints (except `/healthz`) require authentication
- **Auth methods**: Bearer token via `Authorization` header, or session cookie after login
- **CSRF protection**: All POST endpoints require a CSRF token (from session or header)
- **Rate limiting**: Token-bucket rate limiter on DM/guild message sends
- **IP hashing**: Source IPs are SHA-256 hashed (truncated) in audit logs
- **Content redaction**: Sensitive values are redacted in audit log previews
- **No secrets in UI**: Auth tokens, API keys, and full exception traces are never exposed
- **Permission checks**: Guild/channel sends verify bot has `send_messages` permission
- **DM scope**: Only bot-participant DMs are archived; no scraping of private user data

## Endpoints

### Public
- `GET /healthz` — Basic health check (enabled/running state)

### Auth
- `POST /api/login` — Login with auth token (returns session cookie + CSRF token)
- `POST /api/logout` — Logout (clears session)

### API (authenticated)
- `GET /api/summary` — Bot overview stats
- `GET /api/metrics` — Basic metrics
- `GET /api/guilds` — Guild inventory (paginated, searchable)
- `GET /api/guilds/{guild_id}` — Guild detail with channels
- `GET /api/audit` — Audit log (paginated, filterable)
- `GET /api/dms` — DM thread list
- `GET /api/dms/{user_id}` — DM thread messages
- `POST /api/dms/{user_id}/send` — Send DM as bot (CSRF required)
- `POST /api/guilds/{guild_id}/channels/{channel_id}/send` — Send guild message (CSRF required)
- `GET /api/csrf-token` — Get CSRF token for current session

### Static
- `GET /` — Dashboard HTML shell
- `GET /static/{filename}` — CSS/JS assets

## Audit Event Types

| Event | Description |
|---|---|
| `dashboard.login.success` | Successful login |
| `dashboard.login.failure` | Failed login attempt |
| `dashboard.logout` | Logout |
| `dashboard.view.dms` | DM viewer accessed |
| `dashboard.send.dm` | DM sent as bot |
| `dashboard.send.guild_message` | Guild message sent as bot |
| `dashboard.guild.join` | Bot joined a guild |
| `dashboard.guild.leave` | Bot left a guild |
| `dashboard.command.invoke` | Command used via dashboard |
| `dashboard.alert.send` | Alert sent via dashboard |
| `dashboard.config.reload` | Config reloaded via dashboard |
| `dashboard.start` | Dashboard server started |
| `dashboard.stop` | Dashboard server stopped |

## Architecture

```
bot/dashboard/
├── __init__.py          # Package exports
├── config.py            # Environment variable loading
├── server.py            # aiohttp AppRunner lifecycle
├── routes.py            # HTTP route handlers
├── auth.py              # Auth middleware, session store, CSRF
├── audit_store.py       # SQLite audit log (WAL, retention, pagination)
├── dm_store.py          # SQLite DM archive (WAL, retention, pagination)
├── services.py          # Business logic (rate limiting, send operations)
└── static/
    ├── index.html       # Dashboard HTML shell
    ├── dashboard.css    # Styling
    └── dashboard.js     # Client-side JS (vanilla, no frameworks)
```

### Startup Flow

1. `LLMBot.setup_hook()` → `setup_background_tasks()`
2. `_start_dashboard()` background task loads config
3. If `DASHBOARD_ENABLED=true`: creates AuditStore, DMStore, DashboardServices, DashboardServer
4. aiohttp AppRunner starts on configured host:port
5. Shutdown: `LLMBot.close()` calls `dashboard.stop()`

### Integration Points

- Dashboard observes bot state through `DashboardServices` (narrow interface)
- DM archiving hooks into `on_message` for DMChannel messages
- Audit store is separate from other SQLite databases
- No circular imports: dashboard imports from bot core, not vice versa
