# Migration Guide: From Monolithic to Production-Ready Package

This guide explains how to migrate from the original monolithic `main.py` structure to the new production-ready package structure.

## Overview of Changes

### Before (Monolithic Structure)
```
discord-llm-chatbot/
├── main.py (3,000+ lines)
├── requirements.txt
├── .env
└── user_profiles/
```

### After (Production-Ready Package)
```
discord-llm-chatbot/
├── bot/                    # Main package directory
│   ├── __init__.py        # Package exports and metadata
│   ├── main.py            # Bootstrap entry point (<200 LOC)
│   ├── config.py          # Configuration management
│   ├── logs.py            # Structured logging
│   ├── memory.py          # Memory management
│   ├── events.py          # Event handlers
│   ├── commands/          # Command modules
│   │   ├── __init__.py
│   │   ├── memory_cmds.py
│   │   └── tts_cmds.py
│   └── [other modules...]
├── pyproject.toml         # Modern Python packaging
├── MANIFEST.in            # Distribution manifest
├── .env-sample            # Environment documentation
├── CHANGELOG.md           # Change documentation
└── MIGRATION_GUIDE.md     # This file
```

## Migration Steps

### 1. Environment Setup
```bash
# Create and activate virtual environment with uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package in development mode
uv pip install -e .
```

### 2. Configuration Migration

#### Old `.env` format:
```bash
DISCORD_TOKEN=your_token
MEMORY_SAVE_INTERVAL=30 # How often to save profiles
```

#### New `.env` format:
```bash
# ===== DISCORD BOT CONFIGURATION =====
DISCORD_TOKEN=your_token
COMMAND_PREFIX=!

# ===== AI BACKEND CONFIGURATION =====
TEXT_BACKEND=openai
OPENAI_API_KEY=your_key
OPENAI_API_BASE=https://openrouter.ai/api/v1
OPENAI_TEXT_MODEL=deepseek/deepseek-chat-v3-0324:free

# ===== MEMORY & CONVERSATION MANAGEMENT =====
MEMORY_SAVE_INTERVAL=300
```

**Key Changes:**
- Comments in environment variables are now properly handled
- All configuration is documented in `.env-sample`
- OpenRouter integration is fully configured
- Robust parsing with `_safe_int()` and `_safe_float()` helpers

### 3. Execution Changes

#### Old execution:
```bash
python main.py
```

#### New execution:
```bash
# Standard execution
uv run python -m bot.main

# With debug logging
uv run python -m bot.main --debug

# Configuration validation
uv run python -m bot.main --config-check

# Version information
uv run python -m bot.main --version
```

### 4. Import Changes

#### Old imports (if extending the bot):
```python
from main import some_function
```

#### New imports:
```python
from bot import LLMBot, load_config, configure_logging
from bot.memory import get_profile, save_profile
from bot.utils import send_chunks
```

### 5. Package Installation

#### Development installation:
```bash
uv pip install -e .
```

#### Production installation:
```bash
uv pip install -r requirements.txt
```

#### With optional dependencies:
```bash
uv pip install -e ".[dev]"     # Development tools
uv pip install -e ".[test]"    # Testing framework
uv pip install -e ".[all]"     # All optional dependencies
```

## New Features

### 1. Enhanced CLI Interface
```bash
# Show help
uv run python -m bot.main --help

# Debug mode
uv run python -m bot.main --debug

# Set log level
uv run python -m bot.main --log-level DEBUG

# Validate configuration
uv run python -m bot.main --config-check
```

### 2. Comprehensive Error Handling
- **Discord-specific errors**: Login failures, HTTP errors
- **Configuration errors**: Missing tokens, invalid values
- **Import errors**: Missing dependencies, circular imports
- **Graceful degradation**: Continues operation when possible

### 3. Structured Logging
- **File and console handlers**: Separate log levels
- **Configurable output**: Set via CLI or environment
- **Performance logging**: Track bot performance
- **Error tracking**: Comprehensive error reporting

### 4. Package Management
- **Modern packaging**: Uses `pyproject.toml` instead of `setup.py`
- **Dependency separation**: Dev, test, and production dependencies
- **Version management**: Automatic version handling
- **Distribution ready**: Can be published to PyPI

## Troubleshooting

### Common Issues

#### 1. Import Errors
```python
# Error: ModuleNotFoundError: No module named 'bot'
# Solution: Install the package in development mode
uv pip install -e .
```

#### 2. Configuration Errors
```bash
# Error: ValueError: invalid literal for int()
# Solution: Check environment variables for comments
uv run python -m bot.main --config-check
```

#### 3. Discord Connection Issues
```bash
# Error: Invalid Discord token
# Solution: Verify DISCORD_TOKEN in .env file
uv run python -m bot.main --config-check
```

#### 4. Missing Dependencies
```bash
# Error: ModuleNotFoundError: No module named 'trafilatura'
# Solution: Install missing dependencies
uv pip install -r requirements.txt
```

### Environment Issues

#### Virtual Environment Problems
```bash
# Ensure using correct environment
uv venv
uv pip install -e .
```

#### Python Version Compatibility
```bash
# Check Python version (requires 3.9-3.11)
python --version

# If wrong version, install correct Python
# Then recreate virtual environment
uv venv --python 3.11
```

## Validation Checklist

After migration, verify everything works:

- [ ] **Configuration**: `uv run python -m bot.main --config-check`
- [ ] **Version**: `uv run python -m bot.main --version`
- [ ] **Help**: `uv run python -m bot.main --help`
- [ ] **Package import**: `uv run python -c "import bot; print(bot.__version__)"`
- [ ] **Dependencies**: All packages install without errors
- [ ] **Bot startup**: Bot starts without crashing (may not connect without token)

## Performance Improvements

### 1. Reduced Memory Usage
- **Modular loading**: Only load needed components
- **Lazy imports**: Import on demand where appropriate
- **Optimized imports**: Remove unused imports

### 2. Faster Startup
- **Bootstrap design**: Minimal main.py for quick startup
- **Parallel loading**: Background tasks start asynchronously
- **Efficient configuration**: Single config load with caching

### 3. Better Error Recovery
- **Graceful degradation**: Continue operation when possible
- **Automatic retries**: Built-in retry mechanisms
- **Resource cleanup**: Proper cleanup on shutdown

## Support

If you encounter issues during migration:

1. **Check the logs**: Enable debug logging with `--debug`
2. **Validate configuration**: Use `--config-check` to verify settings
3. **Review dependencies**: Ensure all packages are installed
4. **Check Python version**: Must be 3.9-3.11
5. **Use uv environment**: Always use `uv` for environment management

## Next Steps

After successful migration:

1. **Update deployment scripts** to use new execution method
2. **Review configuration** for any new settings
3. **Test all features** to ensure compatibility
4. **Update documentation** for your specific use case
5. **Consider CI/CD** setup using the new package structure

## Router and Testing Refactor

The `Router` class and the corresponding test suite have been significantly refactored to improve reliability, testability, and maintainability.

### Key Changes

1.  **Dependency Injection in Router**: The `Router` now accepts an optional `flow_overrides` dictionary in its constructor. This allows for the injection of mock flow methods during testing, completely isolating the router's logic from the actual implementation of the flows.

2.  **Robust Test Fixtures**: The test suite (`tests/test_router.py`) now uses a set of powerful fixtures to streamline test creation:
    *   `mock_bot`: Provides a fully-mocked `LLMBot` instance with all necessary attributes (`config`, `tts_manager`, `loop`, etc.) to prevent `AttributeError` during tests.
    *   `mock_logger`: Injects a mock logger to prevent side effects and allow for logging assertions.
    *   `router_factory`: A factory fixture that simplifies `Router` instantiation with custom `flow_overrides`.

3.  **`MockParsedCommand` Dataclass**: A `MockParsedCommand` dataclass has replaced the use of `MagicMock` for `ParsedCommand` objects, providing better type safety and clarity in tests.

4.  **Graceful Error Handling**: The `dispatch_message` method now wraps the initial `parse_command` call in a `try...except` block. This ensures that messages that are not valid commands (e.g., regular chat in a guild without a bot mention) are silently ignored, preventing test failures and extraneous error logging.

### Writing a New Router Test (Runbook)

Here is a complete example of how to write a test for the refactored router. This serves as a runbook for future test development.

```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from bot.router import ResponseMessage
from bot.types import Command

# Assuming fixtures like mock_bot, mock_logger, and mock_message are defined as in test_router.py

@pytest.mark.asyncio
async def test_example_flow(router_setup, mock_message):
    """Example test for a custom flow."""
    # 1. Setup: Get the router factory and mocks from the router_setup fixture
    router_factory, mock_parse_command, mock_logger = router_setup

    # 2. Mock the specific flow method you want to test
    mock_custom_flow = AsyncMock(return_value="Custom flow response")

    # 3. Instantiate the router using the factory, providing the mock flow
    router = router_factory(flow_overrides={'process_text': mock_custom_flow})

    # 4. Set up the mock message and the return value for the command parser
    mock_message.content = "!custom_command"
    mock_parse_command.return_value = MockParsedCommand(
        command=Command.CHAT,  # Or any other relevant command
        cleaned_content="custom_command"
    )

    # 5. Dispatch the message
    response = await router.dispatch_message(mock_message)

    # 6. Assert the results
    mock_custom_flow.assert_called_once_with("custom_command", str(mock_message.author.id))
    assert response.text == "Custom flow response"
    assert response.audio_path is None
```
