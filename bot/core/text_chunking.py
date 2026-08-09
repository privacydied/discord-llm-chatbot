"""Single source of truth for splitting outbound text to Discord's size limit. [CA][CMV]

Discord rejects message content over 2000 characters with HTTP 400 / JSON error
50035 ("Must be 2000 or fewer in length"). Every outbound send path must respect
that, and must do so *identically* -- there were previously three splitters in the
tree with different policies (``LLMBot._chunk_message_content`` at 1950 chars,
``bot.utils.send_in_chunks`` at 2000 with no headroom for a mention prefix, and a
1024-char embed-field splitter), which meant the guarantee depended on which code
path a reply happened to take.

The reassembly contract matters as much as the limit: ``"".join(split_for_discord(t)) == t``
for any ``t``. Splitters that re-join on ``"\\n\\n"`` silently mutate content.

Callers are responsible for sending the parts **in order, one await at a time** --
Discord assigns snowflakes in receipt order, so concurrent sends can land a short
trailing part ahead of the long part it continues.
"""

from __future__ import annotations

# Discord's hard limit is 2000; 1950 leaves headroom for a mention prefix or a
# part-marker the caller may prepend after splitting. [CMV]
DISCORD_MAX_CONTENT_LEN = 1950

# Discord JSON error codes seen on the send path. [CMV]
DISCORD_ERR_UNKNOWN_MESSAGE = 10008
DISCORD_ERR_INVALID_FORM_BODY = 50035


def split_for_discord(content: str, max_len: int = DISCORD_MAX_CONTENT_LEN) -> list[str]:
    """Split a text payload into Discord-safe chunks.

    Requirements:
    - Preserve all original content ("".join(chunks) == text).
    - Prefer paragraph boundaries, then line breaks, then sentence endings,
      then whitespace; hard cut only as last resort within max_len.
    - Avoid empty/whitespace-only chunks where possible without dropping content.
    - Respect code-fence parity when choosing a split point if feasible.
    """
    try:
        if content is None:
            return []
        text = str(content)
    except (TypeError, ValueError):
        text = content or ""

    if not text:
        return []

    if len(text) <= max_len:
        return [text]

    chunks: list[str] = []
    n = len(text)
    start = 0

    while start < n:
        remaining = n - start
        if remaining <= max_len:
            # Final chunk: take everything left; preserves exact content.
            chunks.append(text[start:])
            break

        window = text[start : start + max_len]

        # Candidate split positions (relative to start), in priority order.
        candidates: list[int] = []

        para_idx = window.rfind("\n\n")
        if para_idx != -1:
            candidates.append(para_idx + 2)

        line_idx = window.rfind("\n")
        if line_idx != -1:
            candidates.append(line_idx + 1)

        sentence_idx = max(
            window.rfind(". "),
            window.rfind("! "),
            window.rfind("? "),
        )
        if sentence_idx != -1:
            candidates.append(sentence_idx + 2)

        space_idx = max(window.rfind(" "), window.rfind("\t"))
        if space_idx != -1:
            candidates.append(space_idx + 1)

        # Deduplicate while preserving order.
        seen: set[int] = set()
        uniq_candidates: list[int] = []
        for c in candidates:
            if c not in seen and c > 0:
                seen.add(c)
                uniq_candidates.append(c)

        best_break: int | None = None

        # Prefer boundaries that keep us outside of fenced code blocks when possible.
        for candidate in sorted(uniq_candidates, reverse=True):
            segment = text[start : start + candidate]
            if segment.count("```") % 2 == 0:
                best_break = candidate
                break

        # If none chosen by code-fence logic, take the last viable candidate.
        if best_break is None:
            best_break = max(uniq_candidates) if uniq_candidates else max_len

        # Safety clamp: never exceed max_len.
        if best_break <= 0 or (start + best_break > n):
            best_break = min(max_len, n - start)

        # Take this chunk as-is (no rstrip/trimming that loses content).
        chunk = text[start : start + best_break]

        # If the chunk is empty/whitespace-only and there's more content ahead,
        # extend it by one character to avoid producing an effectively empty message.
        if not chunk.strip() and (start + best_break < n):
            extra = min(1, n - (start + best_break))
            chunk = text[start : start + best_break + extra]

        chunks.append(chunk)
        start += best_break

    return chunks
