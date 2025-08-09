"""
Edit coalescing system for router speed overhaul. [REH][CA]

Implements edit coalescing with ≥ EDIT_COALESCE_MIN_MS between edits.
Silences text-only flows to reduce chat noise while preserving streaming for heavy work.

Key requirements:
- Minimum interval between edits (default 700ms)
- Silence text-only flows (no streaming status cards)
- Preserve streaming for media/heavy work
- One-message discipline: only one Discord message per flow
- Coalesce multiple edit attempts into single update
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Dict, Optional, Any, Callable, Awaitable
from dataclasses import dataclass
from enum import Enum

import discord
from bot.config import get_config


class StreamingEligibility(Enum):
    """Streaming eligibility reasons. [CA]"""
    ELIGIBLE = "eligible"
    TEXT_ONLY = "text_only" 
    LIGHT_WORK = "light_work"
    DISABLED = "disabled"


@dataclass
class EditCoalesceState:
    """State tracking for edit coalescing. [CA]"""
    message_id: int
    channel_id: int
    last_edit_time: float
    edit_count: int = 0
    pending_content: Optional[str] = None
    pending_embed: Optional[discord.Embed] = None
    coalesce_task: Optional[asyncio.Task] = None
    is_streaming_eligible: bool = False
    streaming_reason: StreamingEligibility = StreamingEligibility.TEXT_ONLY


class EditCoalescer:
    """Manages edit coalescing for Discord messages. [REH][CA]"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize edit coalescer. [CA]"""
        self.config = config or get_config()
        self.min_interval_ms = float(self.config.get('EDIT_COALESCE_MIN_MS', 700))
        self.max_coalesce_delay_ms = float(self.config.get('EDIT_COALESCE_MAX_DELAY_MS', 2000))
        
        # Track active edit states by message ID
        self.active_states: Dict[int, EditCoalesceState] = {}
        
        # Statistics
        self.stats = {
            'edits_requested': 0,
            'edits_coalesced': 0,
            'edits_sent': 0,
            'text_only_silenced': 0,
            'streaming_preserved': 0
        }
    
    def should_enable_streaming(self, 
                              is_heavy_work: bool,
                              is_video_processing: bool, 
                              is_ocr_processing: bool,
                              has_attachments: bool,
                              estimated_duration_ms: float) -> tuple[bool, StreamingEligibility]:
        """Determine if streaming should be enabled. [CA]"""
        # Text-only flows are never streaming eligible
        if not (is_heavy_work or is_video_processing or is_ocr_processing or has_attachments):
            return False, StreamingEligibility.TEXT_ONLY
        
        # Light work (< 5s estimated) doesn't need streaming
        if estimated_duration_ms < 5000:
            return False, StreamingEligibility.LIGHT_WORK
        
        # Heavy work enables streaming
        if is_heavy_work or is_video_processing or is_ocr_processing:
            return True, StreamingEligibility.ELIGIBLE
        
        # Default to not streaming
        return False, StreamingEligibility.DISABLED
    
    async def start_streaming_session(self,
                                    message: discord.Message,
                                    is_heavy_work: bool = False,
                                    is_video_processing: bool = False,
                                    is_ocr_processing: bool = False,
                                    has_attachments: bool = False,
                                    estimated_duration_ms: float = 0) -> Optional[EditCoalesceState]:
        """Start a streaming session with edit coalescing. [CA]"""
        # Check streaming eligibility
        streaming_eligible, reason = self.should_enable_streaming(
            is_heavy_work, is_video_processing, is_ocr_processing, 
            has_attachments, estimated_duration_ms
        )
        
        # For text-only flows, silence streaming entirely
        if reason == StreamingEligibility.TEXT_ONLY:
            self.stats['text_only_silenced'] += 1
            return None
        
        # Create edit state
        state = EditCoalesceState(
            message_id=message.id,
            channel_id=message.channel.id,
            last_edit_time=time.time(),
            is_streaming_eligible=streaming_eligible,
            streaming_reason=reason
        )
        
        self.active_states[message.id] = state
        
        if streaming_eligible:
            self.stats['streaming_preserved'] += 1
        
        return state
    
    async def request_edit(self,
                          message_id: int,
                          content: Optional[str] = None,
                          embed: Optional[discord.Embed] = None,
                          edit_callback: Optional[Callable[[str, Optional[discord.Embed]], Awaitable[None]]] = None) -> bool:
        """Request an edit with coalescing. [REH]"""
        self.stats['edits_requested'] += 1
        
        state = self.active_states.get(message_id)
        if not state:
            # No active session, edit immediately
            if edit_callback:
                await edit_callback(content, embed)
                self.stats['edits_sent'] += 1
            return True
        
        # Check if enough time has passed since last edit
        now = time.time()
        time_since_last = (now - state.last_edit_time) * 1000  # Convert to ms
        
        if time_since_last >= self.min_interval_ms:
            # Enough time passed, edit immediately
            state.last_edit_time = now
            state.edit_count += 1
            
            if edit_callback:
                await edit_callback(content, embed)
                self.stats['edits_sent'] += 1
            return True
        
        # Need to coalesce - store pending content
        state.pending_content = content
        state.pending_embed = embed
        
        # Cancel existing coalesce task if any
        if state.coalesce_task and not state.coalesce_task.done():
            state.coalesce_task.cancel()
        
        # Schedule coalesced edit
        delay_ms = self.min_interval_ms - time_since_last
        delay_s = delay_ms / 1000.0
        
        state.coalesce_task = asyncio.create_task(
            self._execute_coalesced_edit(message_id, delay_s, edit_callback)
        )
        
        self.stats['edits_coalesced'] += 1
        return False
    
    async def _execute_coalesced_edit(self,
                                    message_id: int,
                                    delay_s: float,
                                    edit_callback: Optional[Callable[[str, Optional[discord.Embed]], Awaitable[None]]]):
        """Execute a coalesced edit after delay. [REH]"""
        try:
            await asyncio.sleep(delay_s)
            
            state = self.active_states.get(message_id)
            if not state or not state.pending_content:
                return
            
            # Execute the edit
            if edit_callback:
                await edit_callback(state.pending_content, state.pending_embed)
                self.stats['edits_sent'] += 1
            
            # Update state
            state.last_edit_time = time.time()
            state.edit_count += 1
            state.pending_content = None
            state.pending_embed = None
            
        except asyncio.CancelledError:
            # Task was cancelled, likely due to newer edit request
            pass
        except Exception as e:
            # Log error but don't crash
            import logging
            logging.error(f"Edit coalescer error: {e}")
    
    async def finalize_session(self, message_id: int) -> None:
        """Finalize a streaming session and cleanup. [RM]"""
        state = self.active_states.get(message_id)
        if not state:
            return
        
        # Cancel any pending coalesce task
        if state.coalesce_task and not state.coalesce_task.done():
            state.coalesce_task.cancel()
            try:
                await state.coalesce_task
            except asyncio.CancelledError:
                pass
        
        # Clean up state
        del self.active_states[message_id]
    
    async def force_flush_pending(self, message_id: int,
                                edit_callback: Optional[Callable[[str, Optional[discord.Embed]], Awaitable[None]]] = None) -> bool:
        """Force flush any pending edits immediately. [REH]"""
        state = self.active_states.get(message_id)
        if not state or not state.pending_content:
            return False
        
        # Cancel coalesce task
        if state.coalesce_task and not state.coalesce_task.done():
            state.coalesce_task.cancel()
        
        # Execute pending edit immediately
        if edit_callback:
            await edit_callback(state.pending_content, state.pending_embed)
            self.stats['edits_sent'] += 1
        
        # Clear pending state
        state.pending_content = None
        state.pending_embed = None
        state.last_edit_time = time.time()
        state.edit_count += 1
        
        return True
    
    def get_session_stats(self, message_id: int) -> Optional[Dict[str, Any]]:
        """Get statistics for a streaming session. [PA]"""
        state = self.active_states.get(message_id)
        if not state:
            return None
        
        return {
            'message_id': state.message_id,
            'channel_id': state.channel_id,
            'edit_count': state.edit_count,
            'streaming_eligible': state.is_streaming_eligible,
            'streaming_reason': state.streaming_reason.value,
            'has_pending': state.pending_content is not None,
            'session_duration_s': time.time() - state.last_edit_time
        }
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get global edit coalescing statistics. [PA]"""
        return {
            **self.stats,
            'active_sessions': len(self.active_states),
            'min_interval_ms': self.min_interval_ms,
            'max_coalesce_delay_ms': self.max_coalesce_delay_ms,
            'efficiency_ratio': (
                self.stats['edits_coalesced'] / max(self.stats['edits_requested'], 1)
            ) * 100
        }


# Global edit coalescer instance
_edit_coalescer: Optional[EditCoalescer] = None


def get_edit_coalescer() -> EditCoalescer:
    """Get global edit coalescer instance. [CA]"""
    global _edit_coalescer
    if _edit_coalescer is None:
        _edit_coalescer = EditCoalescer()
    return _edit_coalescer


async def create_streaming_message(channel: discord.abc.Messageable,
                                 initial_content: str,
                                 is_heavy_work: bool = False,
                                 is_video_processing: bool = False,
                                 is_ocr_processing: bool = False,
                                 has_attachments: bool = False,
                                 estimated_duration_ms: float = 0) -> tuple[discord.Message, Optional[EditCoalesceState]]:
    """Create a message with streaming session if eligible. [CA]"""
    # Send initial message
    message = await channel.send(initial_content)
    
    # Start streaming session if eligible
    coalescer = get_edit_coalescer()
    session = await coalescer.start_streaming_session(
        message, is_heavy_work, is_video_processing, 
        is_ocr_processing, has_attachments, estimated_duration_ms
    )
    
    return message, session


async def update_streaming_message(message: discord.Message,
                                 content: str,
                                 embed: Optional[discord.Embed] = None) -> bool:
    """Update a streaming message with edit coalescing. [REH]"""
    coalescer = get_edit_coalescer()
    
    async def edit_callback(content: str, embed: Optional[discord.Embed]):
        await message.edit(content=content, embed=embed)
    
    return await coalescer.request_edit(
        message.id, content, embed, edit_callback
    )


async def finalize_streaming_message(message: discord.Message,
                                   final_content: str,
                                   final_embed: Optional[discord.Embed] = None) -> None:
    """Finalize a streaming message and cleanup session. [RM]"""
    coalescer = get_edit_coalescer()
    
    # Force flush any pending edits with final content
    async def final_edit_callback(content: str, embed: Optional[discord.Embed]):
        await message.edit(content=final_content, embed=final_embed)
    
    await coalescer.force_flush_pending(message.id, final_edit_callback)
    await coalescer.finalize_session(message.id)


# Export main functions
__all__ = [
    'EditCoalescer',
    'StreamingEligibility', 
    'EditCoalesceState',
    'get_edit_coalescer',
    'create_streaming_message',
    'update_streaming_message', 
    'finalize_streaming_message'
]


if __name__ == "__main__":
    # Demo edit coalescing
    import asyncio
    from unittest.mock import AsyncMock
    
    async def demo():
        coalescer = EditCoalescer({'EDIT_COALESCE_MIN_MS': 500})
        
        # Mock edit callback
        edit_callback = AsyncMock()
        
        print("Testing edit coalescing...")
        
        # Simulate rapid edits
        await coalescer.request_edit(123, "Edit 1", edit_callback=edit_callback)
        await coalescer.request_edit(123, "Edit 2", edit_callback=edit_callback) 
        await coalescer.request_edit(123, "Edit 3", edit_callback=edit_callback)
        
        # Wait for coalescing
        await asyncio.sleep(1.0)
        
        stats = coalescer.get_global_stats()
        print(f"Global stats: {stats}")
        
        await coalescer.finalize_session(123)
    
    asyncio.run(demo())
