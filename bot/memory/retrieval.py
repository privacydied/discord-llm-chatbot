"""Retrieval helpers for curated long-term memory."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .service import build_memory_prompt_block, get_memory_service


async def get_relevant_memories(
    *,
    user_id: Optional[str],
    guild_id: Optional[str],
    channel_id: Optional[str],
    thread_id: Optional[str],
    query: str,
    top_k: int = 6,
) -> List[Dict[str, Any]]:
    # Low-resource top_k cap [Phase 6-9]
    from ..config import load_config as _retrieval_load_config
    _rc = _retrieval_load_config()
    lr_top_k = int(_rc.get("MEMORY_LOW_RESOURCE_TOP_K", top_k))
    effective_top_k = min(top_k, lr_top_k)
    effective_top_k = max(1, effective_top_k)

    service = await get_memory_service()
    return await service.semantic_search(
        query,
        user_id=user_id,
        guild_id=guild_id,
        channel_id=channel_id,
        thread_id=thread_id,
        top_k=effective_top_k,
    )


async def build_relevant_memory_block(**kwargs: Any) -> str:
    return await build_memory_prompt_block(**kwargs)
