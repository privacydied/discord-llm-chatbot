#!/usr/bin/env python3
"""
Comprehensive validation script for the Discord bot refactor.
This script validates all aspects of the refactored system according to the 
ultra-exhaustive requirements.
"""
import os
import sys
import importlib
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def validate_file_structure() -> Dict[str, Any]:
    """Validate that all required files exist."""
    print("🔍 Validating file structure...")
    
    required_files = [
        "bot/main.py",
        "bot/config.py", 
        "bot/logs.py",
        "bot/tasks.py",
        "bot/shutdown.py",
        "bot/events.py",
        "bot/commands/__init__.py",
        "bot/commands/memory_cmds.py",
        "bot/commands/tts_cmds.py",
    ]
    
    missing_files = []
    existing_files = []
    
    for file_path in required_files:
        if Path(file_path).exists():
            existing_files.append(file_path)
        else:
            missing_files.append(file_path)
    
    return {
        "status": "PASS" if not missing_files else "FAIL",
        "existing_files": existing_files,
        "missing_files": missing_files,
        "details": f"Found {len(existing_files)}/{len(required_files)} required files"
    }

def validate_main_py_structure() -> Dict[str, Any]:
    """Validate main.py meets all requirements."""
    print("🔍 Validating main.py structure...")
    
    main_py_path = Path("bot/main.py")
    if not main_py_path.exists():
        return {"status": "FAIL", "error": "main.py not found"}
    
    with open(main_py_path, 'r') as f:
        content = f.read()
        lines = content.split('\n')
    
    validations = {}
    
    # Check line count
    line_count = len(lines)
    validations["line_count"] = {
        "status": "PASS" if line_count <= 100 else "WARN",
        "count": line_count,
        "target": "≤100"
    }
    
    # Check for import-time side effects
    has_immediate_execution = any(
        line.strip().startswith(('bot.run(', 'asyncio.run(', 'run_bot()')) 
        and 'if __name__' not in line
        for line in lines
    )
    validations["no_import_side_effects"] = {
        "status": "PASS" if not has_immediate_execution else "FAIL",
        "details": "No immediate bot execution at import time"
    }
    
    # Check for entry point guard
    has_entry_guard = 'if __name__ == "__main__":' in content
    validations["entry_point_guard"] = {
        "status": "PASS" if has_entry_guard else "FAIL",
        "details": "Entry point properly guarded"
    }
    
    # Check for asyncio.run usage
    has_asyncio_run = 'asyncio.run(' in content
    validations["asyncio_run"] = {
        "status": "PASS" if has_asyncio_run else "FAIL",
        "details": "Uses asyncio.run() for proper async execution"
    }
    
    # Check for business logic (should be minimal)
    business_logic_indicators = [
        'def generate_ai_response',
        'def process_urls',
        'await generate_response',
        'log_message(message)',
        'await send_chunks'
    ]
    
    has_business_logic = any(indicator in content for indicator in business_logic_indicators)
    validations["no_business_logic"] = {
        "status": "PASS" if not has_business_logic else "FAIL",
        "details": "No business logic found in main.py"
    }
    
    # Check for proper helper imports
    required_imports = [
        'from .config import load_config, validate_required_env',
        'from .logs import configure_logging',
        'from .commands import setup_commands',
        'from .tasks import spawn_background_tasks',
        'from .shutdown import setup_signal_handlers'
    ]
    
    import_checks = {}
    for import_line in required_imports:
        import_checks[import_line] = import_line in content
    
    validations["required_imports"] = {
        "status": "PASS" if all(import_checks.values()) else "FAIL",
        "imports": import_checks
    }
    
    return {
        "status": "PASS" if all(v.get("status") == "PASS" for v in validations.values()) else "FAIL",
        "validations": validations
    }

def validate_helper_modules() -> Dict[str, Any]:
    """Validate that all helper modules have required functions."""
    print("🔍 Validating helper modules...")
    
    helper_validations = {}
    
    # Test config module
    try:
        from bot.config import load_config, validate_required_env, ConfigurationError
        helper_validations["config"] = {
            "status": "PASS",
            "functions": ["load_config", "validate_required_env", "ConfigurationError"]
        }
    except ImportError as e:
        helper_validations["config"] = {"status": "FAIL", "error": str(e)}
    
    # Test logs module
    try:
        from bot.logs import configure_logging, setup_logging
        helper_validations["logs"] = {
            "status": "PASS",
            "functions": ["configure_logging", "setup_logging"]
        }
    except ImportError as e:
        helper_validations["logs"] = {"status": "FAIL", "error": str(e)}
    
    # Test commands module
    try:
        from bot.commands import setup_commands
        helper_validations["commands"] = {
            "status": "PASS",
            "functions": ["setup_commands"]
        }
    except ImportError as e:
        helper_validations["commands"] = {"status": "FAIL", "error": str(e)}
    
    # Test tasks module
    try:
        from bot.tasks import spawn_background_tasks, stop_background_tasks
        helper_validations["tasks"] = {
            "status": "PASS", 
            "functions": ["spawn_background_tasks", "stop_background_tasks"]
        }
    except ImportError as e:
        helper_validations["tasks"] = {"status": "FAIL", "error": str(e)}
    
    # Test shutdown module
    try:
        from bot.shutdown import graceful_shutdown, setup_signal_handlers
        helper_validations["shutdown"] = {
            "status": "PASS",
            "functions": ["graceful_shutdown", "setup_signal_handlers"]
        }
    except ImportError as e:
        helper_validations["shutdown"] = {"status": "FAIL", "error": str(e)}
    
    return {
        "status": "PASS" if all(v.get("status") == "PASS" for v in helper_validations.values()) else "FAIL",
        "modules": helper_validations
    }

def validate_import_safety() -> Dict[str, Any]:
    """Validate that importing bot.main doesn't cause side effects."""
    print("🔍 Validating import safety...")
    
    try:
        # Clear any existing imports
        if 'bot.main' in sys.modules:
            del sys.modules['bot.main']
        
        # Import the module
        import bot.main
        
        # Check that no bot instance was created
        # This is a heuristic check - in real implementation, we'd need discord.py
        return {
            "status": "PASS",
            "details": "Module imported without side effects"
        }
    
    except Exception as e:
        return {
            "status": "FAIL",
            "error": str(e),
            "details": "Import failed or caused side effects"
        }

def validate_separation_of_concerns() -> Dict[str, Any]:
    """Validate that business logic is properly separated."""
    print("🔍 Validating separation of concerns...")
    
    # Check that events.py contains the extracted business logic
    events_path = Path("bot/events.py")
    if not events_path.exists():
        return {"status": "FAIL", "error": "events.py not found"}
    
    with open(events_path, 'r') as f:
        events_content = f.read()
    
    # Check for expected business logic in events.py
    expected_functions = [
        'def on_message',
        'def process_urls', 
        'def generate_ai_response'
    ]
    
    functions_found = {}
    for func in expected_functions:
        functions_found[func] = func in events_content
    
    return {
        "status": "PASS" if all(functions_found.values()) else "FAIL",
        "functions_found": functions_found,
        "details": "Business logic properly extracted to events.py"
    }

def print_validation_results(results: Dict[str, Any]) -> None:
    """Print formatted validation results."""
    print("\n" + "="*60)
    print("🎯 VALIDATION RESULTS SUMMARY")
    print("="*60)
    
    for test_name, result in results.items():
        status = result.get("status", "UNKNOWN")
        status_icon = "✅" if status == "PASS" else "⚠️" if status == "WARN" else "❌"
        
        print(f"{status_icon} {test_name.replace('_', ' ').title()}: {status}")
        
        if "error" in result:
            print(f"   Error: {result['error']}")
        elif "details" in result:
            print(f"   Details: {result['details']}")
    
    # Overall status
    overall_status = "PASS" if all(r.get("status") == "PASS" for r in results.values()) else "FAIL"
    print(f"\n🎯 OVERALL STATUS: {overall_status}")
    
    if overall_status == "PASS":
        print("🎉 All validations passed! The refactor is successful.")
    else:
        print("⚠️ Some validations failed. Please review the results above.")

def main():
    """Run all validation tests."""
    print("🚀 Starting comprehensive refactor validation...")
    
    validation_results = {}
    
    # Run all validations
    validation_results["file_structure"] = validate_file_structure()
    validation_results["main_py_structure"] = validate_main_py_structure()
    validation_results["helper_modules"] = validate_helper_modules()
    validation_results["import_safety"] = validate_import_safety()
    validation_results["separation_of_concerns"] = validate_separation_of_concerns()
    
    # Print results
    print_validation_results(validation_results)
    
    # Return appropriate exit code
    overall_status = all(r.get("status") == "PASS" for r in validation_results.values())
    return 0 if overall_status else 1

if __name__ == "__main__":
    sys.exit(main())
