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


# Cap on a captured fence language tag (e.g. "python") used when reopening a
# fence split across chunks. Guards against treating an un-terminated fence
# (no newline before the next max_len window) as an enormous "language". [CMV][REH]
_MAX_FENCE_LANG_LEN = 20


def _apply_fence_toggles(text: str, fence_open: bool, fence_lang: str) -> tuple[bool, str]:
    """Walk every ``` occurrence in ``text`` in order, toggling fence state.

    Shared by the splitter (to prefer break points outside a fence) and by
    ``fence_wrap_markers`` (to decorate parts for rendering). Language tag is
    captured only on the open transition, from the text between the fence
    marker and the next newline. [CMV]
    """
    idx = 0
    while True:
        pos = text.find("```", idx)
        if pos == -1:
            break
        if not fence_open:
            eol = text.find("\n", pos + 3)
            tag_end = eol if eol != -1 else len(text)
            fence_lang = text[pos + 3 : tag_end].strip()[:_MAX_FENCE_LANG_LEN]
            fence_open = True
        else:
            fence_open = False
            fence_lang = ""
        idx = pos + 3
    return fence_open, fence_lang


def split_for_discord(content: str, max_len: int = DISCORD_MAX_CONTENT_LEN) -> list[str]:
    """Split a text payload into Discord-safe chunks.

    Requirements:
    - Preserve all original content ("".join(chunks) == text).
    - Prefer paragraph boundaries, then line breaks, then sentence endings,
      then whitespace; hard cut only as last resort within max_len.
    - Avoid empty/whitespace-only chunks where possible without dropping content.
    - Respect code-fence parity when choosing a split point if feasible.

    This function's own output stays byte-exact and undecorated -- it never
    inserts fence markers, so ``"".join(split_for_discord(t)) == t`` always
    holds. A part that lands inside an open code fence still renders broken on
    its own; ``fence_wrap_markers`` / ``render_chunks_for_discord`` below are
    the presentation layer that fixes that at send time, kept separate so the
    splitter's reassembly guarantee is never in tension with rendering. [CA]
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
    # Cumulative fence state as of `start`, carried across iterations so parity
    # checks account for a fence left open by an earlier chunk -- not just the
    # candidate window in isolation. [REH]
    fence_open = False
    fence_lang = ""

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

        # Prefer boundaries that leave us OUTSIDE a fenced code block, evaluated
        # cumulatively (fence_open carried in) rather than assuming each window
        # starts unfenced. [REH]
        for candidate in sorted(uniq_candidates, reverse=True):
            segment = text[start : start + candidate]
            candidate_open, _ = _apply_fence_toggles(segment, fence_open, fence_lang)
            if not candidate_open:
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
        # Never past max_len, though -- best_break can already equal max_len (an
        # all-whitespace window with no other candidate), and extending further
        # would break the "every part fits the limit" guarantee. [REH]
        if not chunk.strip() and (start + best_break < n) and best_break < max_len:
            extra = min(1, n - (start + best_break))
            best_break += extra
            chunk = text[start : start + best_break]

        chunks.append(chunk)
        fence_open, fence_lang = _apply_fence_toggles(chunk, fence_open, fence_lang)
        start += best_break

    return chunks


def fence_wrap_markers(chunks: list[str]) -> list[tuple[str, str]]:
    """For each raw ``split_for_discord`` part, the (prefix, suffix) that makes
    it render as valid Markdown on its own -- reopening a fence left open by
    the previous part, and closing one this part leaves open.

    Deliberately separate from the split itself: apply these AFTER any text
    sanitization of the chunk body, so the markers bracket the final sent text
    rather than risk a sanitizer mangling them. The wrapped result does NOT
    preserve ``"".join(...) == original`` -- it is a presentation step for
    what actually gets sent, not the splitter's reassembly contract. [CA][REH]
    """
    markers: list[tuple[str, str]] = []
    fence_open = False
    fence_lang = ""
    for chunk in chunks:
        prefix = f"```{fence_lang}\n" if fence_open else ""
        fence_open, fence_lang = _apply_fence_toggles(chunk, fence_open, fence_lang)
        suffix = "\n```" if fence_open else ""
        markers.append((prefix, suffix))
    return markers


def render_chunks_for_discord(chunks: list[str]) -> list[str]:
    """Convenience wrapper: apply ``fence_wrap_markers`` directly to raw parts.

    For callers with no separate per-part sanitization step to interleave with.
    """
    return [f"{prefix}{chunk}{suffix}" for chunk, (prefix, suffix) in zip(chunks, fence_wrap_markers(chunks), strict=True)]
