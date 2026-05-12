from __future__ import annotations

import discord

CAPABILITY_LINES = [
    "Chat: ask anything in plain language.",
    "Images / vision: upload an image or screenshot for analysis.",
    "Screenshot: send a URL with `!ss <url>` to capture + analyze.",
    "Audio / video transcription: upload media or share a supported link.",
    "PDF handling: attach PDFs for summary or Q&A.",
    "URL / web extraction: send a link and I can read it.",
    "Search: use `!search <query>` for web search.",
    "Image gen: use `!img <prompt>`.",
    "TTS: use `!speak <text>` where enabled.",
    "RAG: use `!index <text|url|attachment>`, `!rag status`, `!rag search <query>`, `!rag add <text>`.",
    "Memory: `!memories-show` (your stored memories), `!memory-review`, `!memory-forget <id>`, `!memory-status` (admin).",
    "Memory mgmt: `!memory-add`, `!memory-search <q>`, `!memory-show <id>`, `!memory-export`.",
    "Server archive: `!archive-status`, `!archive-sync`, `!archive-search`, `!archive-pause`/`resume`.",
    "Context: `!context_help`, `!context_reset`, `!context_stats`.",
    "Cleanup: `!clean <target>` (e.g. cache, embeds, msgs), `!clean-help`, `!clean-status`.",
    "Video: `!watch <url>`, `!video-cache`.",
    "Admin / status: `!status`, `!feature <name> <on|off>`, `!reload-config`.",
    "Alert: `!alert` for emergency DM to operators.",
]

CAPABILITY_EXAMPLES = [
    "`!search latest news about Python 3.14`",
    "`!img a neon fox in the rain`",
    "`!speak read this aloud`",
    "`!ss https://example.com`",
    "`!memories-show`",
    "`!memory-forget a1b2c3d4`",
    "`!archive-search what were they talking about on Monday`",
    "`!context_stats`",
    "`!clean cache`",
    "`!watch https://youtu.be/abc123`",
    "`!feature rag on`",
]


def build_help_text() -> str:
    lines = ["Here is what I can do:"]
    lines.extend(f"- {line}" for line in CAPABILITY_LINES)
    lines.append("")
    lines.append("Examples:")
    lines.extend(f"- {example}" for example in CAPABILITY_EXAMPLES)
    return "\n".join(lines)


def build_help_embed() -> discord.Embed:
    embed = discord.Embed(
        title="🤖 Bot Capability Card",
        description="Quick overview of the main things I can do.",
        color=discord.Color.blurple(),
    )
    embed.add_field(
        name="Capabilities (core)",
        value="\n".join(f"• {line}" for line in CAPABILITY_LINES[:9]),
        inline=False,
    )
    embed.add_field(
        name="Capabilities (extended)",
        value="\n".join(f"• {line}" for line in CAPABILITY_LINES[9:]),
        inline=False,
    )
    embed.add_field(
        name="Examples",
        value="\n".join(f"• {example}" for example in CAPABILITY_EXAMPLES),
        inline=False,
    )
    embed.set_footer(text="Tip: attach files or paste a URL when asking for analysis.")
    return embed
