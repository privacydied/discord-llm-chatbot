"""
RAG system configuration management.
"""
import os
from typing import Optional
from .vector_schema import HybridSearchConfig
from ..util.logging import get_logger

logger = get_logger(__name__)


def load_rag_config() -> HybridSearchConfig:
    """
    Load RAG configuration from environment variables.
    
    Returns:
        HybridSearchConfig with values from environment or defaults
    """
    config = HybridSearchConfig(
        # Vector search parameters
        vector_confidence_threshold=float(os.getenv("RAG_VECTOR_CONFIDENCE_THRESHOLD", "0.7")),
        max_vector_results=int(os.getenv("RAG_MAX_VECTOR_RESULTS", "5")),
        vector_weight=float(os.getenv("RAG_VECTOR_WEIGHT", "0.7")),
        
        # Keyword search parameters
        max_keyword_results=int(os.getenv("RAG_MAX_KEYWORD_RESULTS", "3")),
        keyword_weight=float(os.getenv("RAG_KEYWORD_WEIGHT", "0.3")),
        
        # Fallback behavior
        fallback_to_keyword_on_failure=os.getenv("RAG_FALLBACK_ON_FAILURE", "true").lower() == "true",
        fallback_to_keyword_on_low_confidence=os.getenv("RAG_FALLBACK_ON_LOW_CONFIDENCE", "true").lower() == "true",
        min_results_threshold=int(os.getenv("RAG_MIN_RESULTS_THRESHOLD", "1")),
        
        # Result combination
        combine_results=os.getenv("RAG_COMBINE_RESULTS", "true").lower() == "true",
        max_combined_results=int(os.getenv("RAG_MAX_COMBINED_RESULTS", "5")),
        deduplication_threshold=float(os.getenv("RAG_DEDUPLICATION_THRESHOLD", "0.9")),
        
        # Logging and monitoring
        log_retrieval_paths=os.getenv("RAG_LOG_RETRIEVAL_PATHS", "true").lower() == "true",
        log_confidence_scores=os.getenv("RAG_LOG_CONFIDENCE_SCORES", "true").lower() == "true",
        
        # Chunking parameters
        chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "512")),
        chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "50")),
        min_chunk_size=int(os.getenv("RAG_MIN_CHUNK_SIZE", "100")),
        
        # Access control
        enforce_user_scoping=os.getenv("RAG_ENFORCE_USER_SCOPING", "true").lower() == "true",
        enforce_guild_scoping=os.getenv("RAG_ENFORCE_GUILD_SCOPING", "true").lower() == "true",
        
        # Performance & Loading [RAG]
        eager_vector_load=os.getenv("RAG_EAGER_VECTOR_LOAD", "true").lower() == "true",
        background_indexing=os.getenv("RAG_BACKGROUND_INDEXING", "true").lower() == "true",
        
        # Background Processing [RAG]
        indexing_queue_size=int(os.getenv("RAG_INDEXING_QUEUE_SIZE", "1000")),
        indexing_workers=int(os.getenv("RAG_INDEXING_WORKERS", "2")),
        indexing_batch_size=int(os.getenv("RAG_INDEXING_BATCH_SIZE", "10")),
        lazy_load_timeout=float(os.getenv("RAG_LAZY_LOAD_TIMEOUT", "30.0"))
    )
    
    try:
        config.validate()
        logger.debug("[RAG] Configuration loaded and validated successfully")
        return config
    except ValueError as e:
        logger.error(f"[RAG] Invalid configuration: {e}")
        logger.warning("[RAG] Using default configuration")
        return HybridSearchConfig()


def get_rag_environment_info() -> dict:
    """
    Get information about RAG-related environment variables.
    
    Returns:
        Dictionary with environment variable status
    """
    env_vars = {
        # Core RAG settings
        "ENABLE_RAG": os.getenv("ENABLE_RAG", "true"),
        "RAG_DB_PATH": os.getenv("RAG_DB_PATH", "./chroma_db"),
        "RAG_KB_PATH": os.getenv("RAG_KB_PATH", "kb"),
        
        # Embedding model settings
        "RAG_EMBEDDING_MODEL_TYPE": os.getenv("RAG_EMBEDDING_MODEL_TYPE", "sentence-transformers"),
        "RAG_EMBEDDING_MODEL_NAME": os.getenv("RAG_EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"),
        
        # Search parameters
        "RAG_VECTOR_CONFIDENCE_THRESHOLD": os.getenv("RAG_VECTOR_CONFIDENCE_THRESHOLD", "0.7"),
        "RAG_MAX_VECTOR_RESULTS": os.getenv("RAG_MAX_VECTOR_RESULTS", "5"),
        "RAG_VECTOR_WEIGHT": os.getenv("RAG_VECTOR_WEIGHT", "0.7"),
        "RAG_MAX_KEYWORD_RESULTS": os.getenv("RAG_MAX_KEYWORD_RESULTS", "3"),
        "RAG_KEYWORD_WEIGHT": os.getenv("RAG_KEYWORD_WEIGHT", "0.3"),
        
        # Fallback settings
        "RAG_FALLBACK_ON_FAILURE": os.getenv("RAG_FALLBACK_ON_FAILURE", "true"),
        "RAG_FALLBACK_ON_LOW_CONFIDENCE": os.getenv("RAG_FALLBACK_ON_LOW_CONFIDENCE", "true"),
        
        # Chunking settings
        "RAG_CHUNK_SIZE": os.getenv("RAG_CHUNK_SIZE", "512"),
        "RAG_CHUNK_OVERLAP": os.getenv("RAG_CHUNK_OVERLAP", "50"),
        "RAG_MIN_CHUNK_SIZE": os.getenv("RAG_MIN_CHUNK_SIZE", "100"),
        
        # Logging settings
        "RAG_LOG_RETRIEVAL_PATHS": os.getenv("RAG_LOG_RETRIEVAL_PATHS", "true"),
        "RAG_LOG_CONFIDENCE_SCORES": os.getenv("RAG_LOG_CONFIDENCE_SCORES", "true"),
        
        # Performance & Loading settings [RAG]
        "RAG_EAGER_VECTOR_LOAD": os.getenv("RAG_EAGER_VECTOR_LOAD", "true"),
        "RAG_BACKGROUND_INDEXING": os.getenv("RAG_BACKGROUND_INDEXING", "true"),
        
        # Background Processing settings [RAG]
        "RAG_INDEXING_QUEUE_SIZE": os.getenv("RAG_INDEXING_QUEUE_SIZE", "1000"),
        "RAG_INDEXING_WORKERS": os.getenv("RAG_INDEXING_WORKERS", "2"),
        "RAG_INDEXING_BATCH_SIZE": os.getenv("RAG_INDEXING_BATCH_SIZE", "10"),
        "RAG_LAZY_LOAD_TIMEOUT": os.getenv("RAG_LAZY_LOAD_TIMEOUT", "30.0")
    }
    
    return env_vars


def validate_rag_environment() -> tuple[bool, list[str]]:
    """
    Validate RAG environment configuration.
    
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Check if RAG is enabled
    if os.getenv("ENABLE_RAG", "true").lower() != "true":
        return True, ["RAG is disabled"]
    
    # Check required paths
    kb_path = os.getenv("RAG_KB_PATH", "kb")
    if not os.path.exists(kb_path):
        issues.append(f"Knowledge base path does not exist: {kb_path}")
    
    # Check embedding model configuration
    model_type = os.getenv("RAG_EMBEDDING_MODEL_TYPE", "sentence-transformers")
    if model_type not in ["sentence-transformers", "openai"]:
        issues.append(f"Invalid embedding model type: {model_type}")
    
    if model_type == "openai" and not os.getenv("OPENAI_API_KEY"):
        issues.append("OpenAI embedding model selected but OPENAI_API_KEY not set")
    
    # Check numeric parameters
    numeric_params = {
        "RAG_VECTOR_CONFIDENCE_THRESHOLD": (0.0, 1.0),
        "RAG_VECTOR_WEIGHT": (0.0, 1.0),
        "RAG_KEYWORD_WEIGHT": (0.0, 1.0),
        "RAG_DEDUPLICATION_THRESHOLD": (0.0, 1.0),
        "RAG_CHUNK_SIZE": (50, 2048),
        "RAG_CHUNK_OVERLAP": (0, 500),
        "RAG_MIN_CHUNK_SIZE": (10, 1000),
        # Background processing bounds [REH]
        "RAG_INDEXING_QUEUE_SIZE": (1, 10000),
        "RAG_INDEXING_WORKERS": (1, 16),
        "RAG_INDEXING_BATCH_SIZE": (1, 1024),
        # Lazy load timeout bound (0 allows non-blocking path)
        "RAG_LAZY_LOAD_TIMEOUT": (0.0, 600.0),
    }
    
    for param, (min_val, max_val) in numeric_params.items():
        try:
            value = float(os.getenv(param, "0.5"))
            if not min_val <= value <= max_val:
                issues.append(f"{param} must be between {min_val} and {max_val}, got {value}")
        except ValueError:
            issues.append(f"{param} must be a valid number")
    
    # Check weight sum
    try:
        vector_weight = float(os.getenv("RAG_VECTOR_WEIGHT", "0.7"))
        keyword_weight = float(os.getenv("RAG_KEYWORD_WEIGHT", "0.3"))
        if abs(vector_weight + keyword_weight - 1.0) > 1e-6:
            issues.append(f"RAG_VECTOR_WEIGHT + RAG_KEYWORD_WEIGHT must equal 1.0, got {vector_weight + keyword_weight}")
    except ValueError:
        pass  # Already caught above
    
    return len(issues) == 0, issues


# Global configuration instance
_rag_config: Optional[HybridSearchConfig] = None


def get_rag_config() -> HybridSearchConfig:
    """Get the global RAG configuration instance."""
    global _rag_config
    
    if _rag_config is None:
        _rag_config = load_rag_config()
    
    return _rag_config


def reload_rag_config() -> HybridSearchConfig:
    """Reload RAG configuration from environment variables."""
    global _rag_config
    _rag_config = load_rag_config()
    logger.info("[RAG] Configuration reloaded")
    return _rag_config
