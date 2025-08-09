#!/usr/bin/env python3
from bot.enhanced_retry import EnhancedRetryManager
import os

def main():
    mgr = EnhancedRetryManager()
    print("Profiles loaded:")
    for k in ("vision", "text", "media"):
        ladder = mgr.provider_configs.get(k, [])
        print(f"- {k}: {len(ladder)} entries")
        for pc in ladder:
            print(f"  • {pc.name}|{pc.model} (timeout={pc.timeout}s, max_attempts={pc.max_attempts})")
    print("\nBudgets:")
    llm_budget = float(os.environ.get('MULTIMODAL_PER_ITEM_BUDGET', '30.0'))
    media_budget = float(os.environ.get('MEDIA_PER_ITEM_BUDGET', '120.0'))
    print(f"- MULTIMODAL_PER_ITEM_BUDGET (LLM/vision): {llm_budget}s")
    print(f"- MEDIA_PER_ITEM_BUDGET (media): {media_budget}s")

if __name__ == "__main__":
    main()
