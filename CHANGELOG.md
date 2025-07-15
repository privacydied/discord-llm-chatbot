# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-07-13

### Added
- **Production-ready package structure** with proper `__init__.py` files and module organization
- **Comprehensive CLI interface** with `--version`, `--config-check`, `--debug`, and `--log-level` options
- **Modern Python packaging** with `pyproject.toml` and `MANIFEST.in` for distribution
- **Enhanced configuration management** with robust environment variable parsing and validation
- **Structured logging system** with file and console handlers, configurable log levels
- **Cross-platform compatibility** with proper path handling and resource management
- **Comprehensive error handling** with specific Discord error types and graceful degradation
- **OpenRouter integration** with support for deepseek/deepseek-chat-v3-0324:free model
- **Dependency management** with optional dependencies for development, testing, and production
- **Package metadata** with version information, author details, and feature descriptions

### Changed
- **Refactored monolithic main.py** into a bootstrap-only entry point (≤200 LOC)
- **Enhanced main.py** with proper `main()` function, CLI argument parsing, and execution guards
- **Improved import system** with relative imports and proper module exposure
- **Updated configuration** with `_safe_int()` and `_safe_float()` helpers for robust parsing
- **Moved presence setting** from `setup_hook` to `on_ready` to prevent NoneType errors
- **Updated requirements.txt** with all missing dependencies including `trafilatura`
- **Python version compatibility** updated to support Python 3.9-3.11 (required by TTS)

### Fixed
- **Missing dependencies** - Added `trafilatura==1.6.2` for web content extraction
- **Import errors** - Fixed missing `Any` and `discord` imports in `web.py`
- **Configuration parsing** - Resolved `ValueError` for malformed environment variables
- **Circular dependencies** - Validated and confirmed no circular import issues
- **Package installation** - Verified `uv pip install -e .` works correctly
- **CLI execution** - Ensured `uv run python -m bot.main` executes without errors

### Technical Details
- **Entry Point**: `bot.main:run_bot` for CLI execution
- **Package Structure**: Organized by functionality with proper separation of concerns
- **Testing**: Comprehensive import validation and cross-platform compatibility
- **Documentation**: Enhanced docstrings, type hints, and comprehensive environment variable documentation
- **Quality Assurance**: PEP 8 compliance, structured logging, and production-ready error handling

### Migration Guide
- **From monolithic main.py**: The bot now uses a proper package structure with `python -m bot.main`
- **Environment variables**: All configuration is now properly documented in `.env-sample`
- **Dependencies**: Install with `uv pip install -e .` for development or use `pyproject.toml`
- **CLI usage**: Use `uv run python -m bot.main` with optional flags for enhanced functionality

### Security
- **Environment variables**: All sensitive configuration externalized to `.env` files
- **Token validation**: Enhanced Discord token validation with specific error messages
- **Error handling**: Comprehensive error handling prevents information leakage

## [Unreleased]

### Planned
- Unit test suite with pytest
- Integration tests for Discord commands
- Performance monitoring and metrics
- Docker containerization support
- CI/CD pipeline configuration
- Advanced logging with structured JSON output

## [0.2.0] - 2025-07-15

### Fixed
- **Startup Stability**: Resolved all catastrophic startup failures related to `TTSManager` initialization, `KokoroDirect` constructor, asynchronous profile loading, and background task management. The bot is now stable and starts cleanly.
- **TTS Initialization**: Fixed `AttributeError` in `TTSManager` by implementing the `is_available` method in the `KokoroDirect` class.
- **Profile Loading**: Fixed `TypeError` during profile loading by removing incorrect `await` calls on synchronous functions.
- **Background Tasks**: Fixed `AttributeError` in `on_ready` event by removing a redundant and incorrect background task start call.
- **Module Imports**: Fixed `ModuleNotFoundError` for `tts_manager` by removing stale imports and refactoring all TTS commands to use the central `bot.tts_manager` instance.
- **Constructor Arguments**: Fixed `TypeError` on `KokoroDirect` instantiation by removing an invalid `logger` argument.

### Changed
- **TTS Engine**: Replaced the previous TTS implementation with `Kokoro-ONNX` for improved performance and voice quality.
- **TTS Architecture**: Refactored the entire TTS system to be managed by a `TTSManager` class attached to the bot instance, eliminating global state and improving testability.
- **Entry Point**: Changed the recommended run command from `python -m bot.main` to `python run.py` to resolve `RuntimeWarning` and establish a cleaner entry point.

### Added
- **TTS Debugging**: Added comprehensive debug logging for the TTS and router systems to aid in future diagnostics.
- **Error Handling**: Implemented more robust error handling and recovery mechanisms for TTS initialization.
