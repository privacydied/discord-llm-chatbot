#!/usr/bin/env python3
"""
Migration script for Vision Budget Manager [REH][RM]

Safely migrates from float-based budget manager to Money-based system.
Preserves all user data and transaction history.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime, timezone
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.vision.money import Money
from bot.utils.logging import get_logger

logger = get_logger(__name__)


def backup_data(data_dir: Path) -> Path:
    """Create backup of existing data [RM]"""
    backup_dir = data_dir.parent / f"vision_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if data_dir.exists():
        shutil.copytree(data_dir, backup_dir)
        logger.info(f"Created backup at: {backup_dir}")
        return backup_dir
    else:
        logger.warning(f"No existing data directory to backup: {data_dir}")
        return None


def migrate_budgets_file(budgets_file: Path) -> bool:
    """Migrate budgets.json to use Money string format [REH]"""
    if not budgets_file.exists():
        logger.info("No budgets.json to migrate")
        return True
    
    try:
        with open(budgets_file, 'r') as f:
            data = json.load(f)
        
        migrated_count = 0
        for user_id, budget in data.items():
            # Check if already migrated (values are strings)
            if isinstance(budget.get('daily_spent', 0), str):
                logger.info(f"User {user_id} already migrated")
                continue
            
            # Convert float values to Money strings
            money_fields = [
                'daily_limit', 'daily_spent',
                'weekly_limit', 'weekly_spent',
                'monthly_limit', 'monthly_spent',
                'reserved_amount', 'total_spent'
            ]
            
            for field in money_fields:
                if field in budget:
                    # Convert float to Money to ensure proper precision
                    old_value = budget[field]
                    money_value = Money(old_value)
                    budget[field] = money_value.to_json_value()
                    logger.debug(f"  {field}: {old_value} -> {budget[field]}")
            
            # Ensure metadata fields exist
            if 'created_at' not in budget:
                budget['created_at'] = datetime.now(timezone.utc).isoformat()
            if 'updated_at' not in budget:
                budget['updated_at'] = datetime.now(timezone.utc).isoformat()
            
            migrated_count += 1
            logger.info(f"Migrated user {user_id}")
        
        # Write back atomically
        temp_file = budgets_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        
        temp_file.replace(budgets_file)
        
        logger.info(f"Successfully migrated {migrated_count} user budgets")
        return True
        
    except Exception as e:
        logger.error(f"Failed to migrate budgets.json: {e}")
        return False


def verify_migration(data_dir: Path) -> bool:
    """Verify migration was successful [REH]"""
    budgets_file = data_dir / "budgets.json"
    
    if not budgets_file.exists():
        logger.info("No budgets file to verify")
        return True
    
    try:
        with open(budgets_file, 'r') as f:
            data = json.load(f)
        
        for user_id, budget in data.items():
            # Check that money fields are strings
            money_fields = ['daily_spent', 'weekly_spent', 'monthly_spent', 
                          'reserved_amount', 'total_spent']
            
            for field in money_fields:
                if field in budget and not isinstance(budget[field], str):
                    logger.error(f"User {user_id} field {field} is not a string: {type(budget[field])}")
                    return False
            
            # Try to load as Money to verify format
            try:
                for field in money_fields:
                    if field in budget:
                        Money(budget[field])
            except Exception as e:
                logger.error(f"User {user_id} has invalid Money value: {e}")
                return False
        
        logger.info("Migration verification passed")
        return True
        
    except Exception as e:
        logger.error(f"Failed to verify migration: {e}")
        return False


def main():
    """Run migration [REH]"""
    # Determine data directory
    data_dir = Path("data/vision")
    
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])
    
    logger.info(f"Starting Vision Budget Manager migration for: {data_dir}")
    
    # Step 1: Backup existing data
    backup_dir = backup_data(data_dir)
    
    # Step 2: Migrate budgets.json
    budgets_file = data_dir / "budgets.json"
    if not migrate_budgets_file(budgets_file):
        logger.error("Migration failed! Backup preserved at: {backup_dir}")
        sys.exit(1)
    
    # Step 3: Verify migration
    if not verify_migration(data_dir):
        logger.error(f"Migration verification failed! Backup preserved at: {backup_dir}")
        if backup_dir:
            logger.info("To restore backup, run:")
            logger.info(f"  rm -rf {data_dir}")
            logger.info(f"  mv {backup_dir} {data_dir}")
        sys.exit(1)
    
    logger.info("✅ Migration completed successfully!")
    logger.info(f"Backup preserved at: {backup_dir}")
    
    # Update the import in the main codebase
    orchestrator_file = Path("bot/vision/orchestrator.py")
    if orchestrator_file.exists():
        logger.info("\n⚠️  IMPORTANT: Update your code to use the new budget manager:")
        logger.info("  1. In bot/vision/orchestrator.py, change:")
        logger.info("     from bot.vision.budget_manager import VisionBudgetManager")
        logger.info("     to:")
        logger.info("     from bot.vision.budget_manager_v2 import VisionBudgetManager")
        logger.info("  2. Update any other imports of budget_manager")
        logger.info("  3. The new manager uses Money type for all monetary values")


if __name__ == "__main__":
    main()
