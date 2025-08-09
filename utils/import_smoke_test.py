#!/usr/bin/env python3
import sys

mods = [
    'bot.web_extraction_service',
    'bot.router',
    'bot.web',
    'bot.commands.screenshot_commands',
]

failed = False
for m in mods:
    try:
        __import__(m)
        print(f"✔ Imported {m}")
    except Exception as e:
        failed = True
        print(f"✖ Failed to import {m}: {e}", file=sys.stderr)

if failed:
    sys.exit(1)
print("ok")
