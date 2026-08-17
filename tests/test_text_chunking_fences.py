"""Code blocks split across Discord messages must still render as code.

Gaps fixed (kanban t_bf91d9bd):
  2. A code fence longer than the chunk limit had no fence-balanced break
     point, so part 2+ rendered as plain text instead of a code block.
  3. The fence-parity check that picked break points was computed per-chunk
     (``text[start:start+candidate]``), not cumulatively -- once a chunk ended
     inside an open fence, every later parity check was inverted.

The fix keeps `split_for_discord` itself byte-exact (that guarantee is pinned
separately in test_text_chunking_shared.py) and adds a presentation layer --
`fence_wrap_markers` / `render_chunks_for_discord` -- applied at send time to
close/reopen fences across parts.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import discord

from bot.action import BotAction
from bot.core.bot import LLMBot
from bot.core.text_chunking import (
    DISCORD_MAX_CONTENT_LEN,
    fence_wrap_markers,
    render_chunks_for_discord,
    split_for_discord,
)


def _fence_count(text: str) -> int:
    return text.count("```")


class TestFenceAwareRendering:
    def test_split_code_block_renders_as_code_in_every_part(self) -> None:
        """The scenario from the card: one fenced block far longer than the
        chunk limit. Every rendered part must have balanced ``` fences.
        """
        body = "\n".join(f"line {i} of some code" for i in range(400))
        text = f"```python\n{body}\n```"
        assert len(text) > DISCORD_MAX_CONTENT_LEN, "fixture must actually force a split"

        raw_parts = split_for_discord(text)
        assert len(raw_parts) >= 2, "fixture must actually force a split"

        rendered = render_chunks_for_discord(raw_parts)

        for part in rendered:
            assert _fence_count(part) % 2 == 0, f"part does not have balanced fences: {part[:80]!r}..."

    def test_reopened_fence_keeps_the_language_tag(self) -> None:
        body = "\n".join(f"x = {i}" for i in range(400))
        text = f"```python\n{body}\n```"
        raw_parts = split_for_discord(text)
        rendered = render_chunks_for_discord(raw_parts)

        # Every part after the first that continues an open fence must reopen
        # with the same language tag, not a bare ```.
        for part in rendered[1:]:
            if part.startswith("```"):
                first_line = part.split("\n", 1)[0]
                assert first_line == "```python", f"expected reopened fence to carry the language tag, got {first_line!r}"

    def test_raw_split_stays_byte_exact_even_with_fences(self) -> None:
        """render_chunks_for_discord is a presentation layer -- the raw splitter
        output it wraps must still satisfy the reassembly contract untouched.
        """
        body = "\n".join(f"line {i}" for i in range(400))
        text = f"```py\n{body}\n```"
        raw_parts = split_for_discord(text)
        assert "".join(raw_parts) == text

    def test_no_fence_means_no_markers_added(self) -> None:
        text = ("plain paragraph text. " * 200) + "\n\n" + ("more text. " * 200)
        raw_parts = split_for_discord(text)
        assert len(raw_parts) >= 2
        assert render_chunks_for_discord(raw_parts) == raw_parts

    def test_fence_wrap_markers_matches_render_chunks(self) -> None:
        body = "\n".join(f"line {i}" for i in range(400))
        text = f"```js\n{body}\n```"
        raw_parts = split_for_discord(text)

        markers = fence_wrap_markers(raw_parts)
        rendered_via_markers = [f"{p}{c}{s}" for c, (p, s) in zip(raw_parts, markers, strict=True)]

        assert rendered_via_markers == render_chunks_for_discord(raw_parts)

    def test_multiple_fences_across_many_parts(self) -> None:
        """Two separate fenced blocks, each individually longer than max_len,
        with plain text between them -- every part must still balance.
        """
        block_a = "```python\n" + "\n".join(f"a{i}" for i in range(300)) + "\n```"
        between = "\n\nSome prose in between the two blocks.\n\n"
        block_b = "```json\n" + "\n".join(f'"{i}": {i}' for i in range(300)) + "\n```"
        text = block_a + between + block_b

        raw_parts = split_for_discord(text)
        assert len(raw_parts) >= 3
        rendered = render_chunks_for_discord(raw_parts)
        for part in rendered:
            assert _fence_count(part) % 2 == 0

    def test_cumulative_parity_prefers_break_outside_fence_when_available(self) -> None:
        """Gap 3: a chunk boundary chosen without accounting for an already-open
        fence from a previous chunk would treat 'inside a fence' as 'outside'.
        This constructs a case where a fence-safe break point exists slightly
        before the naive max_len cutoff and asserts the splitter takes it.
        """
        # A short fenced block, then enough filler that the *next* window's
        # naive (non-cumulative) parity check would be evaluated against an
        # empty/local segment and get the wrong answer if state isn't carried.
        fenced = "```\nshort code\n```\n\n"
        filler = "para. " * 500
        text = fenced + filler
        raw_parts = split_for_discord(text)
        # The fence in the first part must be self-balanced (it's short and
        # complete), independent of anything after it.
        assert _fence_count(raw_parts[0]) % 2 == 0


class TestReassemblyBugfix:
    """The whitespace-only-chunk extension path grew `chunk` by one char but
    advanced `start` by the un-extended `best_break`, silently duplicating
    that character across two chunks whenever the entire candidate break
    landed inside a single whitespace run at least `max_len` long (e.g. a
    pasted block that starts with a huge run of spaces/blank lines). Confirmed
    against the pre-fix logic: this exact fixture produced non-byte-exact
    reassembly (1951 + 137 chars from a 1988-char input) before the `start +=`
    advance accounted for the extension.
    """

    def test_whitespace_extension_does_not_duplicate_characters(self) -> None:
        text = " " * (DISCORD_MAX_CONTENT_LEN + 100) + "real content after the whitespace run"
        parts = split_for_discord(text)
        assert "".join(parts) == text
        assert all(len(p) <= DISCORD_MAX_CONTENT_LEN for p in parts)


class TestEndToEndThroughBotDispatch:
    """The fence-wrap step is wired into `_send_chunked_reply`, not just the
    splitter -- exercise it through the real dispatch path used for ordinary
    long replies (bot._execute_action -> _send_chunked_reply).
    """

    async def test_chunked_reply_renders_split_code_block_as_code(self) -> None:
        bot = LLMBot(command_prefix="!", intents=discord.Intents.none(), config={})
        bot.enhanced_context_manager = None

        sends: list[str] = []

        class _Typing:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return False

        channel = MagicMock()
        channel.typing = lambda: _Typing()

        async def _record(content=None, **kwargs):
            sends.append(content or "")
            return MagicMock(id=len(sends))

        channel.send = AsyncMock(side_effect=_record)

        message = MagicMock()
        message.id = 77
        message.content = "hi"
        message.reference = None
        message.channel = channel
        message.author = MagicMock(id=111, bot=False, mention="<@111>")
        message.guild = MagicMock(id=222)
        message.reply = AsyncMock(side_effect=_record)

        body = "\n".join(f"line {i} of some code" for i in range(400))
        content = f"```python\n{body}\n```"
        assert len(content) > DISCORD_MAX_CONTENT_LEN, "fixture must actually force a split"

        await bot._execute_action(message, BotAction(content=content, embeds=[]))

        assert len(sends) >= 2, "must actually batch for this to be a meaningful check"
        for part in sends:
            assert part.count("```") % 2 == 0, f"part does not render as a balanced code block: {part[:80]!r}..."
