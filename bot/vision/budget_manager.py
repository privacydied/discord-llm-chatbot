"""
Vision Budget Manager - Cost tracking and quota enforcement

Manages vision generation budgets and spend tracking:
- User and server spending quotas with time-based resets
- Cost estimation and reservation system
- Real-time spend tracking with JSON persistence
- Budget alerts and soft/hard limits
- Detailed spend analytics and reporting

Follows Clean Architecture (CA) and Resource Management (RM) principles.
"""

from __future__ import annotations
import json
import asyncio
import aiofiles
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, NamedTuple
from dataclasses import dataclass, asdict
from enum import Enum

from bot.util.logging import get_logger
from bot.config import load_config
from .types import VisionRequest, VisionProvider, VisionTask

logger = get_logger(__name__)


class BudgetPeriod(Enum):
    """Budget reset periods"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    UNLIMITED = "unlimited"


@dataclass
class BudgetResult:
    """Result of budget validation check"""
    approved: bool
    reason: str
    user_message: str
    remaining_budget: float
    estimated_cost: float
    reset_time: Optional[datetime] = None


@dataclass
class UserBudget:
    """User budget tracking data"""
    user_id: str
    daily_limit: float
    weekly_limit: float
    monthly_limit: float
    daily_spent: float = 0.0
    weekly_spent: float = 0.0
    monthly_spent: float = 0.0
    last_daily_reset: datetime = None
    last_weekly_reset: datetime = None
    last_monthly_reset: datetime = None
    total_spent: float = 0.0
    total_jobs: int = 0
    reserved_amount: float = 0.0
    
    def __post_init__(self):
        if self.last_daily_reset is None:
            self.last_daily_reset = datetime.now(timezone.utc)
        if self.last_weekly_reset is None:
            self.last_weekly_reset = datetime.now(timezone.utc)
        if self.last_monthly_reset is None:
            self.last_monthly_reset = datetime.now(timezone.utc)


@dataclass
class SpendRecord:
    """Individual spending record for audit trail"""
    timestamp: datetime
    user_id: str
    job_id: str
    task: str
    provider: str
    estimated_cost: float
    actual_cost: float
    savings: float  # estimated - actual


class VisionBudgetManager:
    """
    Budget management with quota enforcement and spend tracking
    
    Features:
    - Per-user daily/weekly/monthly quotas
    - Cost estimation and reservation system
    - Real-time spend tracking with rollover protection
    - Budget alerts and notifications
    - Detailed analytics and reporting
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config()
        self.logger = get_logger("vision.budget_manager")
        
        # Budget storage paths
        self.budget_dir = Path(self.config["VISION_DATA_DIR"]) / "budgets"
        self.spend_ledger = Path(self.config["VISION_DATA_DIR"]) / "spend_ledger.jsonl"
        
        # Create directories
        self.budget_dir.mkdir(parents=True, exist_ok=True)
        self.spend_ledger.parent.mkdir(parents=True, exist_ok=True)
        
        # Load budget policies
        self.policy = self._load_budget_policy()
        
        # In-memory cache for performance
        self._budget_cache: Dict[str, UserBudget] = {}
        self._cache_ttl = 300  # 5 minutes
        self._cache_timestamps: Dict[str, datetime] = {}
        
        # Concurrency control
        self._locks: Dict[str, asyncio.Lock] = {}
        
        self.logger.info(f"Vision Budget Manager initialized - budget_dir: {self.budget_dir}, spend_ledger: {self.spend_ledger}, daily_limit: {self.policy.get('default_daily_limit', 10.0)}, monthly_limit: {self.policy.get('default_monthly_limit', 100.0)}")
    
    async def check_budget(self, request: VisionRequest) -> BudgetResult:
        """
        Check if user has sufficient budget for request
        
        Args:
            request: Vision generation request with cost estimate
            
        Returns:
            BudgetResult with approval decision and remaining budget
        """
        user_id = request.user_id
        estimated_cost = request.estimated_cost or 0.1  # Fallback estimate
        
        try:
            async with self._get_user_lock(user_id):
                # Load and update user budget
                budget = await self._load_user_budget(user_id)
                self._reset_expired_periods(budget)
                
                # Check all quota levels
                daily_remaining = budget.daily_limit - (budget.daily_spent + budget.reserved_amount)
                weekly_remaining = budget.weekly_limit - (budget.weekly_spent + budget.reserved_amount)
                monthly_remaining = budget.monthly_limit - (budget.monthly_spent + budget.reserved_amount)
                
                # Find most restrictive limit
                min_remaining = min(daily_remaining, weekly_remaining, monthly_remaining)
                
                # Check if request can be approved
                approved = min_remaining >= estimated_cost
                
                if approved:
                    reason = f"Budget check passed: ${estimated_cost:.3f} within ${min_remaining:.3f} remaining"
                    user_message = ""
                    reset_time = self._get_next_reset_time(budget)
                else:
                    # Determine which limit was hit
                    if daily_remaining < estimated_cost:
                        limit_type = "daily"
                        reset_time = budget.last_daily_reset + timedelta(days=1)
                    elif weekly_remaining < estimated_cost:
                        limit_type = "weekly"  
                        reset_time = budget.last_weekly_reset + timedelta(weeks=1)
                    else:
                        limit_type = "monthly"
                        reset_time = budget.last_monthly_reset + timedelta(days=30)
                    
                    reason = f"{limit_type} budget exceeded: ${estimated_cost:.3f} > ${min_remaining:.3f} remaining"
                    user_message = self._generate_budget_message(limit_type, min_remaining, estimated_cost, reset_time)
                
                result = BudgetResult(
                    approved=approved,
                    reason=reason,
                    user_message=user_message,
                    remaining_budget=min_remaining,
                    estimated_cost=estimated_cost,
                    reset_time=reset_time
                )
                
                # Log result
                if not approved:
                    self.logger.warning(f"Budget limit exceeded - user_id: {user_id}, estimated_cost: {estimated_cost}, remaining: {min_remaining}, daily_spent: {budget.daily_spent}, monthly_spent: {budget.monthly_spent}")
                
                return result
                
        except Exception as e:
            self.logger.error(f"Budget check error - user_id: {user_id}, error: {str(e)}", exc_info=True)
            # Fail safe - allow small requests, block large ones
            if estimated_cost <= 0.50:
                return BudgetResult(
                    approved=True,
                    reason="Budget check failed, allowing small request",
                    user_message="",
                    remaining_budget=1.0,
                    estimated_cost=estimated_cost
                )
            else:
                return BudgetResult(
                    approved=False,
                    reason=f"Budget check failed: {str(e)}",
                    user_message="Unable to verify budget. Please try again.",
                    remaining_budget=0.0,
                    estimated_cost=estimated_cost
                )
    
    async def reserve_budget(self, user_id: str, amount: float) -> None:
        """Reserve budget amount for pending job [CMV]"""
        async with self._get_user_lock(user_id):
            budget = await self._load_user_budget(user_id)
            self._reset_expired_periods(budget)
            
            budget.reserved_amount += amount
            await self._save_user_budget(budget)
            
            self.logger.debug(f"Budget reserved - user_id: {user_id}, amount: {amount}, total_reserved: {budget.reserved_amount}")
    
    async def release_reservation(self, user_id: str, amount: float) -> None:
        """Release reserved budget amount (job cancelled/failed) [CMV]"""
        async with self._get_user_lock(user_id):
            budget = await self._load_user_budget(user_id)
            
            budget.reserved_amount = max(0.0, budget.reserved_amount - amount)
            await self._save_user_budget(budget)
            
            self.logger.debug(f"Budget reservation released - user_id: {user_id}, amount: {amount}, new_reserved: {budget.reserved_amount}")
    
    async def record_actual_cost(self, user_id: str, reserved_amount: float, actual_cost: float) -> None:
        """Record actual job cost and adjust budget [CMV]"""
        async with self._get_user_lock(user_id):
            budget = await self._load_user_budget(user_id)
            self._reset_expired_periods(budget)
            
            # Release reservation and add actual spend
            budget.reserved_amount = max(0.0, budget.reserved_amount - reserved_amount)
            
            budget.daily_spent += actual_cost
            budget.weekly_spent += actual_cost  
            budget.monthly_spent += actual_cost
            budget.total_spent += actual_cost
            budget.total_jobs += 1
            
            await self._save_user_budget(budget)
            
            # Record to spend ledger
            await self._append_spend_record({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user_id": user_id,
                "reserved_amount": reserved_amount,
                "actual_cost": actual_cost,
                "savings": reserved_amount - actual_cost,
                "daily_total": budget.daily_spent,
                "monthly_total": budget.monthly_spent
            })
            
            self.logger.info(f"Actual cost recorded - user_id: {user_id}, actual_cost: {actual_cost}, reserved_amount: {reserved_amount}, daily_total: {budget.daily_spent}, monthly_total: {budget.monthly_spent}")

    async def get_user_budget_status(self, user_id: str) -> Dict[str, Any]:
        """Get detailed budget status for user [CMV]"""
        async with self._get_user_lock(user_id):
            budget = await self._load_user_budget(user_id)
            self._reset_expired_periods(budget)
            
            return {
                "user_id": user_id,
                "daily": {
                    "limit": budget.daily_limit,
                    "spent": budget.daily_spent,
                    "reserved": budget.reserved_amount,
                    "remaining": budget.daily_limit - budget.daily_spent - budget.reserved_amount,
                    "reset_time": (budget.last_daily_reset + timedelta(days=1)).isoformat()
                },
                "weekly": {
                    "limit": budget.weekly_limit,
                    "spent": budget.weekly_spent,
                    "remaining": budget.weekly_limit - budget.weekly_spent - budget.reserved_amount,
                    "reset_time": (budget.last_weekly_reset + timedelta(weeks=1)).isoformat()
                },
                "monthly": {
                    "limit": budget.monthly_limit,
                    "spent": budget.monthly_spent,
                    "remaining": budget.monthly_limit - budget.monthly_spent - budget.reserved_amount,
                    "reset_time": (budget.last_monthly_reset + timedelta(days=30)).isoformat()
                },
                "lifetime": {
                    "total_spent": budget.total_spent,
                    "total_jobs": budget.total_jobs,
                    "average_job_cost": budget.total_spent / max(budget.total_jobs, 1)
                }
            }
    
    async def adjust_user_budget(self, user_id: str, daily_limit: Optional[float] = None, 
                               monthly_limit: Optional[float] = None) -> None:
        """Adjust user budget limits (admin function) [CMV]"""
        async with self._get_user_lock(user_id):
            budget = await self._load_user_budget(user_id)
            
            if daily_limit is not None:
                budget.daily_limit = daily_limit
            if monthly_limit is not None:
                budget.monthly_limit = monthly_limit
                # Weekly is typically 1/4 of monthly
                budget.weekly_limit = monthly_limit * 0.25
            
            await self._save_user_budget(budget)
            
            self.logger.info(f"User budget adjusted - user_id: {user_id}, daily_limit: {budget.daily_limit}, monthly_limit: {budget.monthly_limit}")
    
    async def get_spend_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get spending analytics across all users [CMV]"""
        try:
            if not self.spend_ledger.exists():
                return {"total_spend": 0.0, "total_jobs": 0, "users": 0}
            
            # Read spend ledger
            cutoff_time = datetime.now(timezone.utc) - timedelta(days=days)
            total_spend = 0.0
            total_jobs = 0
            users = set()
            provider_spend = {}
            task_spend = {}
            
            async with aiofiles.open(self.spend_ledger, "r") as f:
                async for line in f:
                    try:
                        record = json.loads(line.strip())
                        record_time = datetime.fromisoformat(record["timestamp"].replace("Z", "+00:00"))
                        
                        if record_time >= cutoff_time:
                            actual_cost = record.get("actual_cost", 0.0)
                            total_spend += actual_cost
                            total_jobs += 1
                            users.add(record.get("user_id"))
                            
                            provider = record.get("provider", "unknown")
                            provider_spend[provider] = provider_spend.get(provider, 0.0) + actual_cost
                            
                            task = record.get("task", "unknown")
                            task_spend[task] = task_spend.get(task, 0.0) + actual_cost
                            
                    except Exception as e:
                        self.logger.debug(f"Failed to parse spend record: {e}")
                        continue
            
            return {
                "period_days": days,
                "total_spend": round(total_spend, 3),
                "total_jobs": total_jobs,
                "unique_users": len(users),
                "average_job_cost": round(total_spend / max(total_jobs, 1), 3),
                "spend_by_provider": provider_spend,
                "spend_by_task": task_spend
            }
            
        except Exception as e:
            self.logger.error(f"Analytics error: {e}")
            return {"error": str(e)}
    
    def _reset_expired_periods(self, budget: UserBudget) -> None:
        """Reset budget periods that have expired [CMV]"""
        now = datetime.now(timezone.utc)
        
        # Reset daily if past midnight UTC
        if now.date() > budget.last_daily_reset.date():
            budget.daily_spent = 0.0
            budget.last_daily_reset = now
        
        # Reset weekly (every Monday)
        if now >= budget.last_weekly_reset + timedelta(weeks=1):
            budget.weekly_spent = 0.0
            budget.last_weekly_reset = now
        
        # Reset monthly (every 30 days)
        if now >= budget.last_monthly_reset + timedelta(days=30):
            budget.monthly_spent = 0.0
            budget.last_monthly_reset = now
    
    def _get_next_reset_time(self, budget: UserBudget) -> datetime:
        """Get next budget reset time [CMV]"""
        daily_reset = budget.last_daily_reset + timedelta(days=1)
        weekly_reset = budget.last_weekly_reset + timedelta(weeks=1)  
        monthly_reset = budget.last_monthly_reset + timedelta(days=30)
        
        return min(daily_reset, weekly_reset, monthly_reset)
    
    def _generate_budget_message(self, limit_type: str, remaining: float, requested: float, reset_time: datetime) -> str:
        """Generate user-friendly budget limit message [CMV]"""
        time_until_reset = reset_time - datetime.now(timezone.utc)
        
        if time_until_reset.total_seconds() < 3600:  # Less than 1 hour
            reset_text = f"in {int(time_until_reset.total_seconds() / 60)} minutes"
        elif time_until_reset.days > 0:
            reset_text = f"in {time_until_reset.days} days"
        else:
            hours = int(time_until_reset.total_seconds() / 3600)
            reset_text = f"in {hours} hours"
        
        return (
            f"💰 **Budget Limit Reached**\n"
            f"Your {limit_type} budget limit has been exceeded.\n\n"
            f"**Current Status:**\n"
            f"• Remaining: ${remaining:.2f}\n" 
            f"• Requested: ${requested:.2f}\n"
            f"• Next reset: {reset_text}\n\n"
            f"💡 **Suggestions:**\n"
            f"• Wait for your budget to reset {reset_text}\n"
            f"• Try a simpler request to reduce costs\n"
            f"• Contact an admin if you need a budget increase"
        )
    
    async def _load_user_budget(self, user_id: str) -> UserBudget:
        """Load user budget from file or create default [CMV]"""
        # Check cache first
        if (user_id in self._budget_cache and 
            user_id in self._cache_timestamps and
            datetime.now(timezone.utc) - self._cache_timestamps[user_id] < timedelta(seconds=self._cache_ttl)):
            return self._budget_cache[user_id]
        
        budget_file = self.budget_dir / f"{user_id}.json"
        
        try:
            if budget_file.exists():
                async with aiofiles.open(budget_file, "r") as f:
                    data = json.loads(await f.read())
                    
                # Convert timestamp strings back to datetime
                for field in ["last_daily_reset", "last_weekly_reset", "last_monthly_reset"]:
                    if field in data and isinstance(data[field], str):
                        data[field] = datetime.fromisoformat(data[field].replace("Z", "+00:00"))
                
                budget = UserBudget(**data)
            else:
                # Create default budget
                budget = UserBudget(
                    user_id=user_id,
                    daily_limit=self.policy.get("default_daily_limit", 5.0),
                    weekly_limit=self.policy.get("default_weekly_limit", 25.0),
                    monthly_limit=self.policy.get("default_monthly_limit", 50.0)
                )
                await self._save_user_budget(budget)
            
            # Update cache
            self._budget_cache[user_id] = budget
            self._cache_timestamps[user_id] = datetime.now(timezone.utc)
            
            return budget
            
        except Exception as e:
            self.logger.error(f"Failed to load budget for {user_id}: {e}")
            # Return default budget on error
            return UserBudget(
                user_id=user_id,
                daily_limit=self.policy.get("default_daily_limit", 5.0),
                weekly_limit=self.policy.get("default_weekly_limit", 25.0),
                monthly_limit=self.policy.get("default_monthly_limit", 50.0)
            )
    
    async def _save_user_budget(self, budget: UserBudget) -> None:
        """Save user budget to file [CMV]"""
        budget_file = self.budget_dir / f"{budget.user_id}.json"
        temp_file = self.budget_dir / f"{budget.user_id}.json.tmp"
        
        try:
            # Convert to dict and handle datetime serialization
            data = asdict(budget)
            for field in ["last_daily_reset", "last_weekly_reset", "last_monthly_reset"]:
                if isinstance(data[field], datetime):
                    data[field] = data[field].isoformat()
            
            # Atomic write
            async with aiofiles.open(temp_file, "w") as f:
                await f.write(json.dumps(data, indent=2))
                await f.flush()
            
            temp_file.rename(budget_file)
            
            # Update cache
            self._budget_cache[budget.user_id] = budget
            self._cache_timestamps[budget.user_id] = datetime.now(timezone.utc)
            
        except Exception as e:
            self.logger.error(f"Failed to save budget for {budget.user_id}: {e}")
            if temp_file.exists():
                temp_file.unlink()
    
    async def _append_spend_record(self, record: Dict[str, Any]) -> None:
        """Append spend record to JSONL ledger [CMV]"""
        try:
            async with aiofiles.open(self.spend_ledger, "a") as f:
                line = json.dumps(record, ensure_ascii=False) + "\n"
                await f.write(line)
                await f.flush()
        except Exception as e:
            self.logger.debug(f"Failed to append spend record: {e}")
    
    def _get_user_lock(self, user_id: str) -> asyncio.Lock:
        """Get or create lock for user [CMV]"""
        if user_id not in self._locks:
            self._locks[user_id] = asyncio.Lock()
        return self._locks[user_id]
    
    def _load_budget_policy(self) -> Dict[str, Any]:
        """Load budget policy from vision policy file [CMV]"""
        try:
            policy_path = Path(self.config["VISION_POLICY_PATH"])
            if policy_path.exists():
                with open(policy_path, "r") as f:
                    policy_data = json.load(f)
                return policy_data.get("budget_manager", {})
            else:
                self.logger.warning(f"Policy file not found: {policy_path}")
                return {}
        except Exception as e:
            self.logger.error(f"Failed to load budget policy: {e}")
            return {}
