"""
Template Caching System - Pre-compile and cache prompt templates for performance.
Implements PA (Performance Awareness) and CMV (Constants over Magic Values) rules.
"""
import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import re
from concurrent.futures import ThreadPoolExecutor

from .phase_constants import PhaseConstants as PC
from .phase_timing import get_timing_manager, PipelineTracker
from ..util.logging import get_logger

logger = get_logger(__name__)

@dataclass
class TemplateMetadata:
    """Metadata for cached template."""
    template_id: str
    file_path: Optional[str] = None
    cached_at: float = field(default_factory=time.time)
    hit_count: int = 0
    last_used: float = field(default_factory=time.time)
    content_hash: str = ""
    variables: List[str] = field(default_factory=list)  # Variables found in template

@dataclass  
class CachedTemplate:
    """Pre-compiled template with optimization."""
    content: str
    metadata: TemplateMetadata
    compiled_sections: Dict[str, str] = field(default_factory=dict)
    static_prefix: str = ""
    static_suffix: str = ""
    variable_count: int = 0
    
    def get_content(self) -> str:
        """Get template content, updating hit count."""
        self.metadata.hit_count += 1
        self.metadata.last_used = time.time()
        return self.content

class TemplateCache:
    """High-performance template cache with pre-compilation."""
    
    def __init__(self, max_cache_size: int = 100):
        self.cache: Dict[str, CachedTemplate] = {}
        self.max_cache_size = max_cache_size
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="template-proc")
        
        # Performance metrics
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "compilations": 0,
            "evictions": 0,
            "avg_compile_time_ms": 0
        }
        
        logger.info("🗂️ TemplateCache initialized")
    
    def _generate_template_id(self, content: str, persona: str = "default") -> str:
        """Generate stable template ID from content and persona."""
        content_hash = hashlib.md5(f"{content}{persona}".encode()).hexdigest()[:8]
        return f"tpl_{persona}_{content_hash}"
    
    def _extract_variables(self, content: str) -> List[str]:
        """Extract template variables from content [PA]."""
        # Find variables like {variable_name}, {{variable}}, or ${variable}
        patterns = [
            r'\{([^}]+)\}',           # {variable}
            r'\{\{([^}]+)\}\}',       # {{variable}}  
            r'\$\{([^}]+)\}',         # ${variable}
        ]
        
        variables = set()
        for pattern in patterns:
            matches = re.findall(pattern, content)
            variables.update(matches)
        
        return list(variables)
    
    def _analyze_template_structure(self, content: str) -> Dict[str, Any]:
        """Analyze template for optimization opportunities [PA]."""
        lines = content.split('\n')
        
        # Find static sections (no variables)
        static_lines = []
        dynamic_lines = []
        
        variable_pattern = re.compile(r'\{[^}]+\}|\{\{[^}]+\}\}|\$\{[^}]+\}')
        
        for line in lines:
            if variable_pattern.search(line):
                dynamic_lines.append(line)
            else:
                static_lines.append(line)
        
        # Identify prefix/suffix static sections
        static_prefix_lines = []
        static_suffix_lines = []
        
        # Find static prefix
        for line in lines:
            if not variable_pattern.search(line):
                static_prefix_lines.append(line)
            else:
                break
        
        # Find static suffix (reverse scan)
        for line in reversed(lines):
            if not variable_pattern.search(line):
                static_suffix_lines.insert(0, line)
            else:
                break
        
        return {
            "total_lines": len(lines),
            "static_lines": len(static_lines),
            "dynamic_lines": len(dynamic_lines),
            "static_prefix": '\n'.join(static_prefix_lines),
            "static_suffix": '\n'.join(static_suffix_lines),
            "optimization_ratio": len(static_lines) / len(lines) if lines else 0
        }
    
    async def _compile_template_async(self, content: str, template_id: str, file_path: Optional[str] = None) -> CachedTemplate:
        """Compile template in thread pool for CPU-intensive work [PA]."""
        loop = asyncio.get_event_loop()
        
        def compile_template():
            start_time = time.time()
            
            # Extract variables and analyze structure
            variables = self._extract_variables(content)
            structure = self._analyze_template_structure(content)
            
            # Create metadata
            metadata = TemplateMetadata(
                template_id=template_id,
                file_path=file_path,
                content_hash=hashlib.md5(content.encode()).hexdigest(),
                variables=variables
            )
            
            # Pre-compile optimized sections [PA]
            compiled_sections = {
                "static_prefix": structure["static_prefix"],
                "static_suffix": structure["static_suffix"],
                "full_content": content
            }
            
            # Create cached template
            cached_template = CachedTemplate(
                content=content,
                metadata=metadata,
                compiled_sections=compiled_sections,
                static_prefix=structure["static_prefix"],
                static_suffix=structure["static_suffix"],
                variable_count=len(variables)
            )
            
            compile_time_ms = int((time.time() - start_time) * 1000)
            
            # Update stats
            self.stats["compilations"] += 1
            old_avg = self.stats["avg_compile_time_ms"]
            self.stats["avg_compile_time_ms"] = (
                (old_avg * (self.stats["compilations"] - 1) + compile_time_ms) /
                self.stats["compilations"]
            )
            
            logger.debug(f"✅ Template compiled: {template_id} ({compile_time_ms}ms, {len(variables)} vars)")
            
            return cached_template
        
        return await loop.run_in_executor(self.executor, compile_template)
    
    def _evict_least_used(self):
        """Evict least recently used templates when cache is full [PA]."""
        if len(self.cache) <= self.max_cache_size:
            return
        
        # Sort by last_used timestamp
        sorted_templates = sorted(
            self.cache.items(),
            key=lambda x: x[1].metadata.last_used
        )
        
        # Remove oldest 20% of cache
        evict_count = max(1, len(self.cache) // 5)
        
        for i in range(evict_count):
            template_id, _ = sorted_templates[i]
            del self.cache[template_id]
            self.stats["evictions"] += 1
        
        logger.debug(f"🗑️ Evicted {evict_count} templates from cache")
    
    async def get_template(
        self, 
        content: Optional[str] = None, 
        file_path: Optional[str] = None,
        persona: str = "default",
        tracker: Optional[PipelineTracker] = None
    ) -> CachedTemplate:
        """Get cached template, compiling if necessary [PA]."""
        
        # Load from file if content not provided
        if content is None and file_path:
            try:
                content = Path(file_path).read_text(encoding='utf-8')
            except Exception as e:
                logger.error(f"❌ Failed to load template from {file_path}: {e}")
                raise
        
        if not content:
            raise ValueError("Either content or valid file_path must be provided")
        
        template_id = self._generate_template_id(content, persona)
        
        # Check cache first [PA]
        if template_id in self.cache:
            self.stats["cache_hits"] += 1
            cached_template = self.cache[template_id]
            
            # Verify content hasn't changed if file-based
            if file_path:
                current_hash = hashlib.md5(content.encode()).hexdigest()
                if current_hash != cached_template.metadata.content_hash:
                    logger.debug(f"🔄 Template changed, recompiling: {template_id}")
                    # Content changed, recompile
                    del self.cache[template_id]
                else:
                    logger.debug(f"✅ Template cache HIT: {template_id} ({cached_template.metadata.hit_count} hits)")
                    return cached_template
        
        # Cache miss - compile template
        self.stats["cache_misses"] += 1
        logger.debug(f"⚠️ Template cache MISS: {template_id}")
        
        # Track compilation phase if tracker provided
        timing_manager = get_timing_manager()
        
        if tracker:
            async with timing_manager.track_phase(
                tracker,
                "TEMPLATE_COMPILE",
                template_id=template_id,
                persona=persona
            ):
                cached_template = await self._compile_template_async(content, template_id, file_path)
        else:
            cached_template = await self._compile_template_async(content, template_id, file_path)
        
        # Add to cache with eviction if needed
        self._evict_least_used()
        self.cache[template_id] = cached_template
        
        return cached_template
    
    async def preload_templates(self, template_files: List[str], personas: List[str] = None):
        """Preload multiple templates for better startup performance [PA]."""
        if personas is None:
            personas = ["default"]
        
        logger.info(f"🔄 Preloading {len(template_files)} templates...")
        start_time = time.time()
        
        # Parallel template loading
        tasks = []
        for file_path in template_files:
            for persona in personas:
                task = self.get_template(file_path=file_path, persona=persona)
                tasks.append(task)
        
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
            load_time = int((time.time() - start_time) * 1000)
            logger.info(f"✅ Templates preloaded in {load_time}ms")
        except Exception as e:
            logger.error(f"❌ Error preloading templates: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
        hit_rate = self.stats["cache_hits"] / total_requests if total_requests > 0 else 0
        
        return {
            "cache_size": len(self.cache),
            "max_cache_size": self.max_cache_size,
            "hit_rate": hit_rate,
            "total_requests": total_requests,
            **self.stats
        }
    
    async def cleanup(self):
        """Clean up resources [RM]."""
        self.executor.shutdown(wait=True)
        self.cache.clear()
        logger.debug("🧹 TemplateCache cleaned up")

class OptimizedPromptBuilder:
    """High-performance prompt builder using cached templates [PA]."""
    
    def __init__(self, template_cache: TemplateCache):
        self.template_cache = template_cache
        self.static_sections_cache: Dict[str, str] = {}  # Cache for deterministic sections
        
    def _build_context_section(
        self,
        user_id: str,
        guild_id: Optional[str],
        history: List[Dict[str, Any]] = None,
        rag_context: str = "",
        server_context: str = ""
    ) -> str:
        """Build context section with caching for static parts [PA]."""
        
        # Create cache key for static context parts
        static_key = f"static_{guild_id or 'dm'}"
        
        if static_key not in self.static_sections_cache:
            # Build static server context part
            static_context = ""
            if server_context:
                static_context += f"Server Context: {server_context}\n\n"
            
            self.static_sections_cache[static_key] = static_context
        
        # Combine static and dynamic parts
        full_context = self.static_sections_cache[static_key]
        
        # Add dynamic parts (history, RAG)
        if rag_context:
            full_context += f"Relevant Context:\n{rag_context}\n\n"
        
        if history:
            # Limit history by token budget [CMV]
            max_tokens = PC.HISTORY_MAX_TOKENS_DM if not guild_id else PC.HISTORY_MAX_TOKENS_GUILD
            
            # Simple token estimation (4 chars ≈ 1 token)
            history_text = ""
            total_chars = 0
            
            for msg in reversed(history[-10:]):  # Last 10 messages max
                msg_text = f"{msg.get('author', 'User')}: {msg.get('content', '')}\n"
                if total_chars + len(msg_text) > max_tokens * 4:
                    break
                history_text = msg_text + history_text
                total_chars += len(msg_text)
            
            if history_text:
                full_context += f"Recent Conversation:\n{history_text}\n"
        
        return full_context
    
    async def build_prompt(
        self,
        user_prompt: str,
        template_file: str,
        user_id: str,
        guild_id: Optional[str] = None,
        persona: str = "default",
        history: List[Dict[str, Any]] = None,
        rag_context: str = "",
        server_context: str = "",
        tracker: Optional[PipelineTracker] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Build optimized prompt using cached templates [PA]."""
        
        # Get cached template
        template = await self.template_cache.get_template(
            file_path=template_file,
            persona=persona,
            tracker=tracker
        )
        
        # Build context section with caching
        context_section = self._build_context_section(
            user_id=user_id,
            guild_id=guild_id,
            history=history,
            rag_context=rag_context,
            server_context=server_context
        )
        
        # Use pre-compiled static sections when possible [PA]
        if template.static_prefix and not any(var in template.static_prefix for var in ['{', '$']):
            # Pure static prefix - use as-is
            system_prompt = template.static_prefix
        else:
            # Dynamic template - do minimal substitution
            system_prompt = template.get_content()
        
        # Add context section
        full_system_prompt = f"{system_prompt}\n\n{context_section}".strip()
        
        # Build messages array
        messages = [
            {"role": "system", "content": full_system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return {
            "messages": messages,
            "template_id": template.metadata.template_id,
            "template_hits": template.metadata.hit_count,
            "variable_count": template.variable_count,
            "context_chars": len(context_section),
            "total_chars": sum(len(msg["content"]) for msg in messages)
        }

# Global template cache instance [PA]
_template_cache_instance: Optional[TemplateCache] = None

def get_template_cache() -> TemplateCache:
    """Get global template cache instance."""
    global _template_cache_instance
    
    if _template_cache_instance is None:
        _template_cache_instance = TemplateCache(max_cache_size=50)
        logger.info("🚀 Global TemplateCache created")
    
    return _template_cache_instance

def get_prompt_builder() -> OptimizedPromptBuilder:
    """Get optimized prompt builder with cached templates."""
    cache = get_template_cache()
    return OptimizedPromptBuilder(cache)

async def cleanup_template_cache():
    """Clean up global template cache [RM]."""
    global _template_cache_instance
    if _template_cache_instance:
        await _template_cache_instance.cleanup()
        _template_cache_instance = None
