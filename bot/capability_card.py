from __future__ import annotations

import discord

CAPABILITY_LINES = [
    "Chat: ask anything in plain language.",
    "Images / vision: upload an image or screenshot for analysis.",
    "Audio / video transcription: upload media or share a supported link.",
    "PDF handling: attach PDFs for summary or Q&A.",
    "URL / web extraction: send a link and I can read it.",
    "Search: use `!search <query>` for web search.",
    "RAG: use `!index <text|url|attachment>` or `!rag status`, `!rag search <query>`, `!rag add <text>`.",
    "TTS: use `!speak <text>` or `!tts` commands where enabled.",
    "Image generation: use `!img <prompt>`.",
    "Memory: use `!memory ...` and `!server-memory ...`.",
    "Admin / status: use `!status`, `!feature <name> <on|off>`, and `!reload-config`.",
]

CAPABILITY_EXAMPLES = [
    "`!search latest news about Python 3.14`",
    "`!img a neon fox in the rain`",
    "`!speak read this aloud`",
    "`!rag add These notes should be indexed for future Q&A`",
    "Upload a PDF, image, audio clip, or paste a URL directly.",
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
        name="Capabilities",
        value="\n".join(f"• {line}" for line in CAPABILITY_LINES),
        inline=False,
    )
    embed.add_field(
        name="Examples",
        value="\n".join(f"• {example}" for example in CAPABILITY_EXAMPLES),
        inline=False,
    )
    embed.set_footer(text="Tip: attach files or paste a URL when asking for analysis.")
    return embed
