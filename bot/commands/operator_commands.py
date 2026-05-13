from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional

import discord
from discord.ext import commands

from ..capability_card import build_help_embed
from ..server_features import (
    FEATURE_DEFAULTS,
    feature_status_emoji,
    feature_status_label,
    get_server_feature_toggles,
    normalize_feature_name,
    set_server_feature_toggle,
)
from ..stt import stt_manager
from ..utils.logging import get_logger

logger = get_logger(__name__)


class OperatorCommands(commands.Cog):
    """Operator-facing help, health, and feature toggle commands."""

    def __init__(self, bot) -> None:
        self.bot = bot
        self._boot_time = getattr(bot, "_boot_time", None)
        if self._boot_time is None:
            self._boot_time = datetime.now(timezone.utc)

    @commands.command(name="help", aliases=["capabilities", "capability"])
    async def help_command(self, ctx: commands.Context) -> None:
        """Show the static capability card."""
        await ctx.reply(embed=build_help_embed(), mention_author=False)

    @commands.command(name="status")
    @commands.has_permissions(administrator=True)
    async def status_command(self, ctx: commands.Context) -> None:
        """Show a lightweight operator health summary."""
        try:
            embed = self._build_status_embed(ctx)
            await ctx.reply(embed=embed, mention_author=False)
        except commands.MissingPermissions:
            await ctx.reply(
                "❌ You need administrator permissions to view bot status.",
                mention_author=False,
            )
        except Exception as exc:
            logger.error("status command failed: %s", exc, exc_info=True)
            await ctx.reply("❌ Failed to build status summary.", mention_author=False)

    @commands.command(name="feature", aliases=["toggle-feature", "toggle_feature"])
    @commands.cooldown(2, 300, type=commands.BucketType.user)
    @commands.has_permissions(administrator=True)
    async def feature_command(
        self, ctx: commands.Context, name: str = "", setting: str = ""
    ) -> None:
        """Toggle a per-server feature flag on or off."""
        try:
            if not name:
                await ctx.reply(
                    "Usage: `!feature <stt|tts|vision|image|web|x|rag> <on|off>`",
                    mention_author=False,
                )
                return

            if not setting:
                await ctx.reply(
                    "Usage: `!feature <stt|tts|vision|image|web|x|rag> <on|off>`",
                    mention_author=False,
                )
                return

            normalized = normalize_feature_name(name)
            if normalized not in FEATURE_DEFAULTS:
                await ctx.reply(
                    "❌ Unknown feature. Try: stt, tts, vision, image, web, x, rag.",
                    mention_author=False,
                )
                return

            setting_norm = setting.strip().lower()
            if setting_norm not in {
                "on",
                "off",
                "enable",
                "disable",
                "enabled",
                "disabled",
            }:
                await ctx.reply(
                    "❌ Setting must be `on` or `off`.", mention_author=False
                )
                return

            enabled = setting_norm in {"on", "enable", "enabled"}
            if ctx.guild is None:
                await ctx.reply(
                    "❌ Feature toggles require a server context.", mention_author=False
                )
                return

            set_server_feature_toggle(ctx.guild.id, normalized, enabled)
            state = feature_status_label(enabled)
            await ctx.reply(
                f"{feature_status_emoji(enabled)} `{normalized}` is now {state} for this server.",
                mention_author=False,
            )
            logger.info(
                "feature.toggle | guild=%s feature=%s enabled=%s",
                ctx.guild.id,
                normalized,
                enabled,
            )
        except commands.MissingPermissions:
            await ctx.reply(
                "❌ You need administrator permissions to toggle features.",
                mention_author=False,
            )
        except Exception as exc:
            logger.error("feature command failed: %s", exc, exc_info=True)
            await ctx.reply("❌ Failed to update feature toggle.", mention_author=False)

    def _build_status_embed(self, ctx: commands.Context) -> discord.Embed:
        guild = ctx.guild
        guild_id = getattr(guild, "id", None)
        toggles = (
            get_server_feature_toggles(guild_id) if guild_id else dict(FEATURE_DEFAULTS)
        )

        uptime = self._format_uptime()
        rss = self._format_rss_mb()
        backend = self._get_backend_name()
        rag_line = self._get_rag_status(guild_id, toggles)
        stt_line = self._get_stt_status(toggles)
        tts_line = self._get_tts_status(toggles)
        ollama_line = self._get_ollama_status(toggles)
        playwright_line = self._get_playwright_status(toggles)
        queue_line = self._get_queue_status()
        storage_line = self._get_storage_status()
        embedding_line = self._get_embedding_model_status()

        vision_line = self._get_vision_status(toggles)
        memory_line = self._get_memory_service_status(toggles)
        degraded_line = self._get_degraded_mode_line()

        embed = discord.Embed(
            title="🩺 Bot Status",
            color=discord.Color.red() if degraded_line else discord.Color.green(),
            timestamp=discord.utils.utcnow(),
        )
        embed.add_field(name="Uptime", value=uptime, inline=True)
        embed.add_field(name="Active backend", value=backend, inline=True)
        if rss:
            embed.add_field(name="Memory RSS", value=rss, inline=True)
        embed.add_field(name="Vision", value=vision_line, inline=False)
        if degraded_line:
            embed.add_field(name="Degraded", value=degraded_line, inline=False)
        embed.add_field(name="Memory service", value=memory_line, inline=False)
        embed.add_field(name="RAG", value=rag_line, inline=False)
        embed.add_field(name="STT", value=stt_line, inline=False)
        embed.add_field(name="TTS", value=tts_line, inline=False)
        embed.add_field(name="Ollama", value=ollama_line, inline=False)
        embed.add_field(name="Playwright", value=playwright_line, inline=False)
        embed.add_field(name="Queue / backpressure", value=queue_line, inline=False)
        embed.add_field(name="Storage", value=storage_line, inline=False)
        embed.add_field(name="Embedding model", value=embedding_line, inline=False)

        feature_lines = [
            f"• {name}: {feature_status_label(enabled)}"
            for name, enabled in sorted(toggles.items())
        ]
        if feature_lines:
            embed.add_field(
                name="Feature toggles", value="\n".join(feature_lines), inline=False
            )

        embed.set_footer(
            text="Health checks are cached/local only; no outbound probes are sent."
        )
        return embed

    def _format_uptime(self) -> str:
        boot_time = self._boot_time
        now = datetime.now(timezone.utc)
        if isinstance(boot_time, (int, float)):
            seconds = max(0, int(now.timestamp() - float(boot_time)))
        elif isinstance(boot_time, datetime):
            if boot_time.tzinfo is None:
                boot_time = boot_time.replace(tzinfo=timezone.utc)
            seconds = max(0, int((now - boot_time).total_seconds()))
        else:
            return "unknown"

        days, rem = divmod(seconds, 86400)
        hours, rem = divmod(rem, 3600)
        minutes, sec = divmod(rem, 60)
        if days:
            return f"{days}d {hours}h {minutes}m"
        if hours:
            return f"{hours}h {minutes}m {sec}s"
        return f"{minutes}m {sec}s"

    def _format_rss_mb(self) -> str:
        try:
            import psutil

            process = psutil.Process()
            rss_mb = process.memory_info().rss / (1024 * 1024)
            return f"{rss_mb:.1f} MB"
        except Exception:
            return "unavailable"

    def _get_backend_name(self) -> str:
        config = getattr(self.bot, "config", {}) or {}
        backend_name = config.get("TEXT_BACKEND", config.get("text_backend", "unknown"))
        model = config.get(
            "OPENAI_TEXT_MODEL", config.get("openai_text_model", "unknown")
        )
        if backend_name == "openrouter":
            return f"openrouter ({model})"
        return f"{backend_name} ({model})"

    def _get_vision_status(self, toggles: Dict[str, bool]) -> str:
        """Return Vision / image-gen status line from existing orchestrator state."""
        enabled = toggles.get("vision", True)
        orchestrator = getattr(self.bot, "_vision_orchestrator", None)
        if orchestrator is None:
            return f"{feature_status_emoji(enabled)} {feature_status_label(enabled)}; orchestrator=missing"

        # Try to get adapter status
        adapter = getattr(orchestrator, "adapter", None) or getattr(
            orchestrator, "unified_adapter", None
        )
        providers = getattr(adapter, "providers", {}) if adapter else {}
        provider_count = len(providers) if providers else 0

        # Check if any providers are configured
        config = getattr(self.bot, "config", {}) or {}
        api_key = config.get("VISION_API_KEY") or config.get("vision_api_key")
        has_key = bool(api_key and api_key.strip())

        parts = [f"{feature_status_emoji(enabled)} {feature_status_label(enabled)}"]
        if provider_count:
            parts.append(f"providers={provider_count}")
        if has_key:
            parts.append("key=configured")
        else:
            parts.append("key=missing")
        return "; ".join(parts)

    def _get_memory_service_status(self, toggles: Dict[str, bool]) -> str:
        """Return memory service status from the existing service state."""
        enabled = toggles.get("memory", True)
        svc = getattr(self.bot, "_memory_service", None) or getattr(
            self.bot, "memory_service", None
        )
        if svc is None:
            return f"{feature_status_emoji(enabled)} {feature_status_label(enabled)}; service=missing"

        # Get existing stats from the memory service
        try:
            status = getattr(svc, "get_status", lambda: {})() or {}
        except Exception:
            status = {}

        memory_count = status.get("memory_count", "?")
        store = getattr(svc, "_persistent_store", None)
        store_type = "chroma" if store else "local"

        return f"{feature_status_emoji(enabled)} {feature_status_label(enabled)}; memories={memory_count}; store={store_type}"

    def _get_degraded_mode_line(self) -> str:
        """Return degraded mode info from the existing metrics system, or None."""
        try:
            from bot.metrics import is_degraded_mode, get_degraded_reasons
        except ImportError:
            return None

        if not is_degraded_mode():
            return None

        reasons = get_degraded_reasons()
        if reasons:
            reason_text = "; ".join(str(r) for r in reasons if r)
            return f"⚠️ yes — {reason_text}" if reason_text else "⚠️ yes"
        return "⚠️ yes"

    def _get_rag_status(self, guild_id: Optional[int], toggles: Dict[str, bool]) -> str:
        global_enabled = bool(
            (getattr(self.bot, "config", {}) or {}).get("rag_enabled", True)
        )
        guild_enabled = toggles.get("rag", True) if guild_id is not None else True
        effective = global_enabled and guild_enabled
        try:
            from ..rag import hybrid_search as rag_hybrid_search

            initialized = getattr(rag_hybrid_search, "_hybrid_search", None) is not None
        except Exception:
            initialized = False
        return f"{feature_status_emoji(effective)} {feature_status_label(effective)} (initialized={initialized})"

    def _get_stt_status(self, toggles: Dict[str, bool]) -> str:
        enabled = toggles.get("stt", True)
        available = getattr(stt_manager, "available", False)
        default_spec = getattr(stt_manager, "default_spec", None)
        spec_text = getattr(default_spec, "size", None)
        model_text = f"model={spec_text}" if spec_text else "model=unknown"
        return f"{feature_status_emoji(enabled)} {feature_status_label(enabled)}; {model_text}; loaded={available}"

    def _get_tts_status(self, toggles: Dict[str, bool]) -> str:
        enabled = toggles.get("tts", True)
        tts_manager = getattr(self.bot, "tts_manager", None)
        if tts_manager is None:
            return f"{feature_status_emoji(enabled)} {feature_status_label(enabled)}; manager=missing"
        engine_status = {}
        try:
            engine_status = tts_manager.get_status() or {}
        except Exception:
            engine_status = {}
        cache_len = len(getattr(tts_manager, "_file_cache", {}) or {})
        cache_max = getattr(tts_manager, "_cache_max", None)
        cache_text = f"cache={cache_len}"
        if cache_max is not None:
            cache_text += f"/{cache_max}"
        available = engine_status.get("available", False)
        engine = engine_status.get("engine", "unknown")
        warmup_status = engine_status.get("warmup_status", "unknown")
        return f"{feature_status_emoji(enabled)} {feature_status_label(enabled)}; engine={engine}; loaded={available}; warmup={warmup_status}; {cache_text}"

    def _get_ollama_status(self, toggles: Dict[str, bool]) -> str:
        """Return Ollama provider status (if configured)."""
        import os

        ollama_host = os.getenv("OLLAMA_HOST", "").strip()
        if not ollama_host:
            return "not configured"
        enabled = toggles.get("ollama", True)
        # Check if an ollama backend exists on the router
        router = getattr(self.bot, "router", None)
        ollama_backend = None
        if router:
            ollama_backend = getattr(router, "ollama_backend", None)
        available = ollama_backend is not None
        return (
            f"{feature_status_emoji(enabled)} {feature_status_label(enabled)}; "
            f"host={ollama_host}; available={available}"
        )

    def _get_playwright_status(self, toggles: Dict[str, bool]) -> str:
        """Return Playwright / web-extraction status including health probe info."""
        try:
            from ..utils.playwright_helpers import _pw_server_url, get_playwright_health
            from ..web_extraction_service import ENABLE_TIER_B

            configured = _pw_server_url() is not None
            health = get_playwright_health()
            hw_available = health.get("available")
            cons_fails = health.get("consecutive_failures", 0)
            degraded = health.get("degraded", False)
            service = getattr(self.bot, "web_extraction_service", None)
            tier_b_available = (
                getattr(service, "_tier_b_available", ENABLE_TIER_B)
                if service
                else ENABLE_TIER_B
            )
            enabled = toggles.get("web_extraction", True)
            health_parts: list[str] = []
            if hw_available is True:
                health_parts.append("health=ok")
            elif hw_available is False:
                label = "degraded" if degraded else "health=fail"
                health_parts.append(label)
            else:
                health_parts.append("health=unchecked")
            if cons_fails:
                health_parts.append(f"fails={cons_fails}")
            health_str = "; ".join(health_parts) if health_parts else "health=n/a"
            return (
                f"{feature_status_emoji(enabled)} {feature_status_label(enabled)}; "
                f"configured={configured}; available={tier_b_available}; {health_str}"
            )
        except Exception:
            return "unknown"

    def _get_queue_status(self) -> str:
        active_tasks = len(getattr(self.bot, "_active_long_running_tasks", {}) or {})
        user_queues = getattr(self.bot, "_user_queues", {}) or {}
        queued_messages = 0
        queue_sizes = []
        for q in user_queues.values():
            try:
                size = q.qsize()
            except Exception:
                size = 0
            queued_messages += size
            queue_sizes.append(size)
        backpressure = "yes" if queued_messages or active_tasks else "no"
        return f"active_tasks={active_tasks}; user_queues={len(user_queues)}; queued={queued_messages}; backpressure={backpressure}"


    def _get_storage_status(self) -> str:
        """Return a concise storage summary: mem_db, chroma, and cache sizes."""
        # Try diagnostics module first
        try:
            from ..maintenance.diagnostics import get_storage_status as _gs
            result = _gs()
            if isinstance(result, str) and len(result) < 300:
                return result
        except Exception:
            pass

        parts: list[str] = []
        # In-memory DB size (if present on bot)
        mem_db = getattr(self.bot, "_mem_db", None) or getattr(self.bot, "mem_db", None)
        if mem_db is not None:
            try:
                get_status_fn = getattr(mem_db, "get_status", None)
                if callable(get_status_fn):
                    status = get_status_fn() or {}
                    if isinstance(status, dict):
                        wal_mb = status.get("wal_size_mb", status.get("wal_mb"))
                        if wal_mb is not None:
                            parts.append(f"wal={wal_mb}MB")
                        db_mb = status.get("db_size_mb", status.get("db_mb"))
                        if db_mb is not None:
                            parts.append(f"mem_db={db_mb}MB")
                        entry_count = status.get("entry_count", status.get("entries"))
                        if entry_count is not None:
                            parts.append(f"entries={entry_count}")
                if isinstance(mem_db, dict) and not parts:
                    parts.append(f"mem_db={len(mem_db)} entries")
            except Exception:
                parts.append("mem_db=?")
        else:
            parts.append("mem_db=none")

        # Chroma persistent store size
        memory_svc = getattr(self.bot, "_memory_service", None) or getattr(
            self.bot, "memory_service", None
        )
        store = getattr(memory_svc, "_persistent_store", None) if memory_svc else None
        if store is not None:
            try:
                persist_dir = getattr(store, "_persist_directory", None) or getattr(
                    store, "persist_directory", None
                )
                if persist_dir:
                    import pathlib

                    total = 0
                    for p in pathlib.Path(persist_dir).rglob("*"):
                        try:
                            if p.is_file():
                                total += p.stat().st_size
                        except Exception:
                            pass
                    parts.append(f"chroma={total / (1024*1024):.1f}MB")
                else:
                    parts.append("chroma=in-memory")
            except Exception:
                parts.append("chroma=?")
        else:
            parts.append("chroma=none")

        # TTS / general cache
        tts_mgr = getattr(self.bot, "tts_manager", None)
        if tts_mgr is not None:
            try:
                cache_stats = tts_mgr.get_cache_stats() or {}
                cache_mb = cache_stats.get("size_mb")
                cache_files = cache_stats.get("files")
                if cache_mb is not None:
                    parts.append(f"cache={cache_mb}MB/{cache_files}f")
                else:
                    parts.append("cache=empty")
            except Exception:
                parts.append("cache=?")

        result = " | ".join(parts)
        if len(result) > 300:
            result = result[:297] + "..."
        return result

    def _get_embedding_model_status(self) -> str:
        """Check if the RAG embedding model is cached locally."""
        import os

        model_name = os.getenv(
            "RAG_EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
        ).strip()
        model_type = os.getenv("RAG_EMBEDDING_MODEL_TYPE", "sentence-transformers").strip()

        if model_type != "sentence-transformers":
            return f"{model_name} (type={model_type}; cache_check=skipped)"

        # Check common huggingface cache locations
        cache_dirs: list[str] = []
        hf_home = os.getenv("HF_HOME", "")
        if hf_home:
            cache_dirs.append(hf_home)
        xf_cache = os.getenv("XDG_CACHE_HOME", "")
        if xf_cache:
            cache_dirs.append(os.path.join(xf_cache, "huggingface", "hub"))
        import pathlib

        cache_dirs.append(str(pathlib.Path.home() / ".cache" / "huggingface" / "hub"))

        # Also check bot-level data dirs
        for candidate in ["data/rag", "data/embeddings"]:
            cache_dirs.append(candidate)

        cached = False
        found_at = None
        for cdir in cache_dirs:
            try:
                safe_name = model_name.replace("/", "--")
                model_path = os.path.join(cdir, f"models--{safe_name}")
                alt_path = os.path.join(cdir, safe_name)
                if os.path.exists(model_path) or os.path.exists(alt_path):
                    cached = True
                    found_at = cdir
                    break
                for root, dirs, files in os.walk(cdir):
                    if safe_name.lower() in root.lower() or model_name.lower() in root.lower():
                        if any(
                            f in files
                            for f in ("model.safetensors", "pytorch_model.bin", "config.json")
                        ):
                            cached = True
                            found_at = root
                            break
                    if cached:
                        break
                if cached:
                    break
            except Exception:
                continue

        if cached:
            loc = found_at or "unknown"
            return f"{model_name} | cached ({loc})"
        return f"{model_name} | not cached (first query will download)"


async def setup(bot) -> None:
    await bot.add_cog(OperatorCommands(bot))
    logger.info("✅ OperatorCommands cog loaded")
