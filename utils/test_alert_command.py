#!/usr/bin/env python3
"""
Test script to verify !alert command registration and routing.
"""


def test_imports():
    """Test if all required components import successfully."""
    print("=== Testing Imports ===")

    try:
        print("✅ AdminAlertCommands imports successfully")
    except Exception as e:
        print(f"❌ AdminAlertCommands import failed: {e}")
        return False

    try:
        from bot.types import Command

        has_alert = hasattr(Command, "ALERT")
        print(f"✅ Command.ALERT exists: {has_alert}")
        if not has_alert:
            print("❌ Command.ALERT is missing from enum!")
            return False
    except Exception as e:
        print(f"❌ Command enum import failed: {e}")
        return False

    try:
        from bot.command_parser import COMMAND_MAP

        alert_in_map = "!alert" in COMMAND_MAP
        print(f"✅ !alert in COMMAND_MAP: {alert_in_map}")
        if alert_in_map:
            print(f"✅ !alert maps to: {COMMAND_MAP['!alert']}")
        else:
            print("❌ !alert is missing from COMMAND_MAP!")
            return False
    except Exception as e:
        print(f"❌ COMMAND_MAP import failed: {e}")
        return False

    return True


def test_cog_registration():
    """Test if AdminAlertCommands is in the module definitions."""
    print("\n=== Testing Cog Registration ===")

    try:
        from bot.core.bot import module_definitions

        # Check if AdminAlertCommands is in module definitions
        admin_alert_found = False
        for module_name, cog_name in module_definitions:
            if cog_name == "AdminAlertCommands":
                admin_alert_found = True
                print(
                    f"✅ AdminAlertCommands found in module_definitions: ({module_name}, {cog_name})"
                )
                break

        if not admin_alert_found:
            print("❌ AdminAlertCommands NOT found in module_definitions!")
            print("Available cogs:")
            for module_name, cog_name in module_definitions:
                print(f"  - {cog_name} from {module_name}")
            return False

    except Exception as e:
        print(f"❌ module_definitions import failed: {e}")
        return False

    return True


def test_command_parser():
    """Test command parser logic."""
    print("\n=== Testing Command Parser ===")

    try:
        from bot.command_parser import parse_command
        from bot.types import Command

        # Test parsing !alert
        result = parse_command("!alert test message", is_dm=True)
        print(f"✅ parse_command('!alert test message', is_dm=True) = {result}")

        if result and result.get("command") == Command.ALERT:
            print("✅ !alert correctly parsed as Command.ALERT")
        else:
            print(f"❌ !alert parsing failed. Expected Command.ALERT, got: {result}")
            return False

    except Exception as e:
        print(f"❌ Command parser test failed: {e}")
        return False

    return True


if __name__ == "__main__":
    print("🔍 Testing !alert command registration...\n")

    all_passed = True
    all_passed &= test_imports()
    all_passed &= test_cog_registration()
    all_passed &= test_command_parser()

    print(f"\n{'=' * 50}")
    if all_passed:
        print("🎉 ALL TESTS PASSED! The !alert command should be working.")
        print("If it's still not working, the issue is likely:")
        print("  1. Bot needs restart to load new cog")
        print("  2. Runtime error in AdminAlertCommands.alert() method")
        print("  3. Discord permissions issue")
    else:
        print("❌ SOME TESTS FAILED! The !alert command registration has issues.")
        print("Check the failed components above.")
