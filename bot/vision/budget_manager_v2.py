"""
Vision Budget Manager with Money type and atomic writes [CA][REH][RM]

Manages user budgets for vision generation with:
- Type-safe Money calculations using Decimal
- Atomic JSON ledger writes (temp + fsync + rename)
- Proper reservation tracking
- Transaction logging for audit trail
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, Any
import asyncio
import json
import os
import tempfile

from bot.vision.money import Money
from bot.vision.types import VisionRequest, VisionProvider
from bot.vision.pricing_loader import get_pricing_table
from bot.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class UserBudget:
    """User budget tracking with Money type [CA]"""

    user_id: str

    # Daily limits and tracking (all Money objects stored as strings for JSON)
    daily_limit: str = "5.00"
    daily_spent: str = "0.00"
    daily_reset_time: Optional[str] = None

    # Weekly limits and tracking
    weekly_limit: str = "20.00"
    weekly_spent: str = "0.00"
    weekly_reset_time: Optional[str] = None

    # Monthly limits and tracking
    monthly_limit: str = "50.00"
    monthly_spent: str = "0.00"
    monthly_reset_time: Optional[str] = None

    # Reserved amounts for pending jobs
    reserved_amount: str = "0.00"

    # Total tracking
    total_spent: str = "0.00"
    total_jobs: int = 0

    # Metadata
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> UserBudget:
        """Create from dict during JSON deserialization"""
        return cls(**data)

    # Money property helpers
    def get_daily_spent(self) -> Money:
        return Money(self.daily_spent)

    def set_daily_spent(self, amount: Money) -> None:
        self.daily_spent = amount.to_json_value()

    def get_weekly_spent(self) -> Money:
        return Money(self.weekly_spent)

    def set_weekly_spent(self, amount: Money) -> None:
        self.weekly_spent = amount.to_json_value()

    def get_monthly_spent(self) -> Money:
        return Money(self.monthly_spent)

    def set_monthly_spent(self, amount: Money) -> None:
        self.monthly_spent = amount.to_json_value()

    def get_reserved_amount(self) -> Money:
        return Money(self.reserved_amount)

    def set_reserved_amount(self, amount: Money) -> None:
        self.reserved_amount = amount.to_json_value()

    def get_total_spent(self) -> Money:
        return Money(self.total_spent)

    def set_total_spent(self, amount: Money) -> None:
        self.total_spent = amount.to_json_value()


@dataclass
class BudgetResult:
    """Result of budget check with Money amounts"""

    approved: bool
    reason: str
    user_message: str
    remaining_budget: Money = Money("0.00")
    estimated_cost: Money = Money("0.00")
    reset_time: Optional[datetime] = None
    # Legacy/extended optional fields for compatibility with older tests/DTOs
    daily_remaining: Optional[Money] = None
    weekly_remaining: Optional[Money] = None
    monthly_remaining: Optional[Money] = None
    daily_reserved: Optional[Money] = None
    weekly_reserved: Optional[Money] = None
    monthly_reserved: Optional[Money] = None

    def __post_init__(self) -> None:
        """Normalize types to Money for robustness [IV][REH].

        Tests may construct BudgetResult using floats/strs for fields or include
        legacy fields (daily_remaining, etc.). We coerce where necessary.
        """
        # Coerce remaining_budget and estimated_cost to Money if needed
        if not isinstance(self.remaining_budget, Money):
            self.remaining_budget = Money(self.remaining_budget)
        if not isinstance(self.estimated_cost, Money):
            self.estimated_cost = Money(self.estimated_cost)

        # Helper to coerce optional fields
        def _to_money_opt(v: Optional[Money]) -> Optional[Money]:
            if v is None:
                return None
            return v if isinstance(v, Money) else Money(v)

        self.daily_remaining = _to_money_opt(self.daily_remaining)
        self.weekly_remaining = _to_money_opt(self.weekly_remaining)
        self.monthly_remaining = _to_money_opt(self.monthly_remaining)
        self.daily_reserved = _to_money_opt(self.daily_reserved)
        self.weekly_reserved = _to_money_opt(self.weekly_reserved)
        self.monthly_reserved = _to_money_opt(self.monthly_reserved)


@dataclass
class TransactionRecord:
    """Transaction log entry for audit trail [REH]"""

    timestamp: str
    user_id: str
    transaction_type: str  # "reserve", "finalize", "release", "reset"
    job_id: Optional[str]
    amount: str  # Money as string
    reserved_amount: str  # Money as string
    actual_amount: Optional[str] = None  # Money as string for finalize
    discrepancy_ratio: Optional[float] = None
    provider: Optional[str] = None
    task: Optional[str] = None
    daily_balance: str = "0.00"
    weekly_balance: str = "0.00"
    monthly_balance: str = "0.00"
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization"""
        return asdict(self)


class VisionBudgetManager:
    """
    Manages vision generation budgets with atomic persistence [CA][REH][RM]

    Features:
    - Type-safe Money calculations
    - Atomic file writes with fsync
    - Transaction logging
    - Proper reservation tracking
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize budget manager with config"""
        self.config = config
        self.data_dir = Path(config.get("VISION_DATA_DIR", "data/vision"))
        self.budgets_file = self.data_dir / "budgets.json"
        self.transactions_file = self.data_dir / "transactions.jsonl"
        self.spend_ledger_file = self.data_dir / "spend_ledger.jsonl"

        # Create data directory if needed
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # User locks for concurrent access
        self._user_locks: Dict[str, asyncio.Lock] = {}
        self._lock_manager = asyncio.Lock()

        # Pricing table
        self.pricing_table = get_pricing_table()

        logger.info(f"VisionBudgetManager initialized with data_dir: {self.data_dir}")

    def _get_user_lock(self, user_id: str) -> asyncio.Lock:
        """Get or create per-user lock [REH]"""
        if user_id not in self._user_locks:
            self._user_locks[user_id] = asyncio.Lock()
        return self._user_locks[user_id]

    def _atomic_write_json(self, file_path: Path, data: Any) -> None:
        """
        Atomic JSON write with temp file + fsync + rename [RM][REH]

        This is SYNCHRONOUS to avoid AsyncTextIOWrapper.fsync errors.
        """
        # Create temp file in same directory for atomic rename
        temp_fd, temp_path = tempfile.mkstemp(
            dir=file_path.parent, prefix=f".{file_path.name}.", suffix=".tmp"
        )

        try:
            # Write JSON to temp file
            with os.fdopen(temp_fd, "w") as f:
                json.dump(data, f, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())  # Force write to disk

            # Atomic rename (on same filesystem)
            os.replace(temp_path, file_path)

            # Sync directory to ensure rename is persisted
            dir_fd = os.open(file_path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)

        except Exception:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    def _append_jsonl(self, file_path: Path, record: Any) -> None:
        """
        Append to JSONL file with fsync [RM][REH]

        This is SYNCHRONOUS to avoid AsyncTextIOWrapper.fsync errors.
        """
        # Ensure file exists
        file_path.touch(exist_ok=True)

        # Convert dataclass to dict if needed
        if hasattr(record, "to_dict"):
            data = record.to_dict()
        elif hasattr(record, "__dict__"):
            data = record.__dict__
        else:
            data = record

        # Append with fsync
        with open(file_path, "a") as f:
            json.dump(data, f, default=str)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())

    async def _load_all_budgets(self) -> Dict[str, UserBudget]:
        """Load all user budgets from JSON [REH]"""
        if not self.budgets_file.exists():
            return {}

        try:
            with open(self.budgets_file, "r") as f:
                data = json.load(f)

            budgets = {}
            for user_id, budget_data in data.items():
                budgets[user_id] = UserBudget.from_dict(budget_data)
            return budgets

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to load budgets: {e}")
            return {}

    async def _save_all_budgets(self, budgets: Dict[str, UserBudget]) -> None:
        """Save all user budgets atomically [RM]"""
        data = {user_id: budget.to_dict() for user_id, budget in budgets.items()}
        self._atomic_write_json(self.budgets_file, data)

    async def _load_user_budget(self, user_id: str) -> UserBudget:
        """Load or create user budget [REH]"""
        budgets = await self._load_all_budgets()

        if user_id not in budgets:
            # Create new budget with defaults from config
            budget = UserBudget(
                user_id=user_id,
                daily_limit=str(self.config.get("VISION_DAILY_LIMIT", 5.0)),
                weekly_limit=str(self.config.get("VISION_WEEKLY_LIMIT", 20.0)),
                monthly_limit=str(self.config.get("VISION_MONTHLY_LIMIT", 50.0)),
            )
            budgets[user_id] = budget
            await self._save_all_budgets(budgets)

        return budgets[user_id]

    async def _save_user_budget(self, budget: UserBudget) -> None:
        """Save user budget atomically [RM]"""
        budget.updated_at = datetime.now(timezone.utc).isoformat()

        budgets = await self._load_all_budgets()
        budgets[budget.user_id] = budget
        await self._save_all_budgets(budgets)

    def _reset_expired_periods(self, budget: UserBudget) -> None:
        """Reset budget periods that have expired [CMV]"""
        now = datetime.now(timezone.utc)

        # Check daily reset
        if budget.daily_reset_time:
            reset_time = datetime.fromisoformat(budget.daily_reset_time)
            if now >= reset_time:
                budget.set_daily_spent(Money.zero())
                budget.daily_reset_time = (
                    (now + timedelta(days=1))
                    .replace(hour=0, minute=0, second=0, microsecond=0)
                    .isoformat()
                )
        else:
            # Initialize reset time
            budget.daily_reset_time = (
                (now + timedelta(days=1))
                .replace(hour=0, minute=0, second=0, microsecond=0)
                .isoformat()
            )

        # Check weekly reset
        if budget.weekly_reset_time:
            reset_time = datetime.fromisoformat(budget.weekly_reset_time)
            if now >= reset_time:
                budget.set_weekly_spent(Money.zero())
                # Next Monday at midnight
                days_until_monday = (7 - now.weekday()) % 7 or 7
                budget.weekly_reset_time = (
                    (now + timedelta(days=days_until_monday))
                    .replace(hour=0, minute=0, second=0, microsecond=0)
                    .isoformat()
                )
        else:
            # Initialize to next Monday
            days_until_monday = (7 - now.weekday()) % 7 or 7
            budget.weekly_reset_time = (
                (now + timedelta(days=days_until_monday))
                .replace(hour=0, minute=0, second=0, microsecond=0)
                .isoformat()
            )

        # Check monthly reset
        if budget.monthly_reset_time:
            reset_time = datetime.fromisoformat(budget.monthly_reset_time)
            if now >= reset_time:
                budget.set_monthly_spent(Money.zero())
                # First day of next month
                if now.month == 12:
                    next_month = now.replace(year=now.year + 1, month=1, day=1)
                else:
                    next_month = now.replace(month=now.month + 1, day=1)
                budget.monthly_reset_time = next_month.replace(
                    hour=0, minute=0, second=0, microsecond=0
                ).isoformat()
        else:
            # Initialize to first of next month
            if now.month == 12:
                next_month = now.replace(year=now.year + 1, month=1, day=1)
            else:
                next_month = now.replace(month=now.month + 1, day=1)
            budget.monthly_reset_time = next_month.replace(
                hour=0, minute=0, second=0, microsecond=0
            ).isoformat()

    async def check_budget(self, request: VisionRequest) -> BudgetResult:
        """
        Check if user has sufficient budget for request [REH][CMV]

        Considers both spent and reserved amounts.
        """
        # Get estimated cost using pricing table
        if request.estimated_cost and request.estimated_cost > 0:
            estimated_cost = Money(request.estimated_cost)
        else:
            # Calculate estimate from pricing table
            estimated_cost = self.pricing_table.estimate_cost(
                provider=(request.preferred_provider or VisionProvider.TOGETHER),
                task=request.task,
                width=request.width,
                height=request.height,
                num_images=getattr(request, "batch_size", 1),
                duration_seconds=getattr(request, "duration_seconds", 4.0),
                model=request.model,
            )

        async with self._get_user_lock(request.user_id):
            budget = await self._load_user_budget(request.user_id)
            self._reset_expired_periods(budget)

            # Calculate remaining budgets (considering reserved amounts)
            daily_limit = Money(budget.daily_limit)
            daily_spent = budget.get_daily_spent()
            daily_reserved = budget.get_reserved_amount()
            daily_committed = daily_spent + daily_reserved
            daily_remaining = (daily_limit - daily_committed).clamp_minimum(0)

            weekly_limit = Money(budget.weekly_limit)
            weekly_spent = budget.get_weekly_spent()
            weekly_committed = weekly_spent + daily_reserved
            weekly_remaining = (weekly_limit - weekly_committed).clamp_minimum(0)

            monthly_limit = Money(budget.monthly_limit)
            monthly_spent = budget.get_monthly_spent()
            monthly_committed = monthly_spent + daily_reserved
            monthly_remaining = (monthly_limit - monthly_committed).clamp_minimum(0)

            # Check against all limits
            min_remaining = min(daily_remaining, weekly_remaining, monthly_remaining)
            approved = min_remaining >= estimated_cost

            # Determine limiting factor
            if not approved:
                if daily_remaining < estimated_cost:
                    reason = "daily_limit_exceeded"
                    reset_time = datetime.fromisoformat(budget.daily_reset_time)
                    user_message = (
                        f"Daily budget limit reached. "
                        f"Remaining: {daily_remaining.to_display_string()}, "
                        f"Required: {estimated_cost.to_display_string()}. "
                        f"Resets in {self._format_time_until(reset_time)}."
                    )
                elif weekly_remaining < estimated_cost:
                    reason = "weekly_limit_exceeded"
                    reset_time = datetime.fromisoformat(budget.weekly_reset_time)
                    user_message = (
                        f"Weekly budget limit reached. "
                        f"Remaining: {weekly_remaining.to_display_string()}, "
                        f"Required: {estimated_cost.to_display_string()}. "
                        f"Resets in {self._format_time_until(reset_time)}."
                    )
                else:
                    reason = "monthly_limit_exceeded"
                    reset_time = datetime.fromisoformat(budget.monthly_reset_time)
                    user_message = (
                        f"Monthly budget limit reached. "
                        f"Remaining: {monthly_remaining.to_display_string()}, "
                        f"Required: {estimated_cost.to_display_string()}. "
                        f"Resets in {self._format_time_until(reset_time)}."
                    )
            else:
                reason = "approved"
                reset_time = None
                user_message = ""

            logger.info(
                f"Budget check - user: {request.user_id}, "
                f"estimated: {estimated_cost}, "
                f"daily_remaining: {daily_remaining}, "
                f"reserved: {daily_reserved}, "
                f"approved: {approved}"
            )

            return BudgetResult(
                approved=approved,
                reason=reason,
                user_message=user_message,
                remaining_budget=min_remaining,
                estimated_cost=estimated_cost,
                reset_time=reset_time,
            )

    async def reserve_budget(self, user_id: str, amount: Money) -> None:
        """Reserve budget for pending job [REH]"""
        async with self._get_user_lock(user_id):
            budget = await self._load_user_budget(user_id)
            current_reserved = budget.get_reserved_amount()
            new_reserved = current_reserved + amount
            budget.set_reserved_amount(new_reserved)
            await self._save_user_budget(budget)

            # Log transaction
            self._append_jsonl(
                self.transactions_file,
                TransactionRecord(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    user_id=user_id,
                    transaction_type="reserve",
                    job_id=None,
                    amount=amount.to_json_value(),
                    reserved_amount=new_reserved.to_json_value(),
                    daily_balance=budget.daily_spent,
                    weekly_balance=budget.weekly_spent,
                    monthly_balance=budget.monthly_spent,
                ),
            )

            logger.info(
                f"Reserved budget - user: {user_id}, "
                f"amount: {amount}, "
                f"total_reserved: {new_reserved}"
            )

    async def release_reservation(self, user_id: str, amount: Money) -> None:
        """Release reserved budget (job cancelled/failed) [REH]"""
        async with self._get_user_lock(user_id):
            budget = await self._load_user_budget(user_id)
            current_reserved = budget.get_reserved_amount()
            # Avoid constructing negative Money to prevent noisy warnings [REH]
            if amount >= current_reserved:
                new_reserved = Money.zero()
            else:
                new_reserved = current_reserved - amount
            budget.set_reserved_amount(new_reserved)
            await self._save_user_budget(budget)

            # Log transaction
            self._append_jsonl(
                self.transactions_file,
                TransactionRecord(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    user_id=user_id,
                    transaction_type="release",
                    job_id=None,
                    amount=amount.to_json_value(),
                    reserved_amount=new_reserved.to_json_value(),
                    daily_balance=budget.daily_spent,
                    weekly_balance=budget.weekly_spent,
                    monthly_balance=budget.monthly_spent,
                ),
            )

            logger.info(
                f"Released reservation - user: {user_id}, "
                f"amount: {amount}, "
                f"total_reserved: {new_reserved}"
            )

    async def record_actual_cost(
        self,
        user_id: str,
        reserved_amount: Money,
        actual_cost: Money,
        provider: Optional[str] = None,
        task: Optional[str] = None,
        job_id: Optional[str] = None,
    ) -> None:
        """
        Record actual job cost and adjust budget [REH][CMV]

        Handles discrepancy checking and capping.
        """
        async with self._get_user_lock(user_id):
            budget = await self._load_user_budget(user_id)
            self._reset_expired_periods(budget)

            # Calculate discrepancy ratio
            discrepancy_ratio = actual_cost.ratio_to(reserved_amount)
            max_ratio = self.pricing_table.get_max_discrepancy_ratio()
            warning_ratio = self.pricing_table.get_warning_discrepancy_ratio()

            # Cap actual cost if discrepancy too high
            capped_cost = actual_cost
            notes = None
            if discrepancy_ratio > max_ratio:
                capped_cost = reserved_amount * float(max_ratio)
                logger.error(
                    f"Cost discrepancy too high - user: {user_id}, "
                    f"estimated: {reserved_amount}, "
                    f"actual: {actual_cost}, "
                    f"ratio: {discrepancy_ratio}, "
                    f"capped_to: {capped_cost}"
                )
                notes = "capped"
            elif discrepancy_ratio > warning_ratio:
                logger.warning(
                    f"Cost discrepancy warning - user: {user_id}, "
                    f"estimated: {reserved_amount}, "
                    f"actual: {actual_cost}, "
                    f"ratio: {discrepancy_ratio}"
                )

            # Release reservation and add actual spend
            current_reserved = budget.get_reserved_amount()
            # Prevent negative intermediate results [REH]
            if reserved_amount >= current_reserved:
                new_reserved = Money.zero()
            else:
                new_reserved = current_reserved - reserved_amount
            budget.set_reserved_amount(new_reserved)

            # Update spent amounts
            daily_spent = budget.get_daily_spent() + capped_cost
            budget.set_daily_spent(daily_spent)

            weekly_spent = budget.get_weekly_spent() + capped_cost
            budget.set_weekly_spent(weekly_spent)

            monthly_spent = budget.get_monthly_spent() + capped_cost
            budget.set_monthly_spent(monthly_spent)

            total_spent = budget.get_total_spent() + capped_cost
            budget.set_total_spent(total_spent)

            budget.total_jobs += 1

            await self._save_user_budget(budget)

            # Log transaction
            self._append_jsonl(
                self.transactions_file,
                TransactionRecord(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    user_id=user_id,
                    transaction_type="finalize",
                    job_id=job_id,
                    amount=reserved_amount.to_json_value(),
                    reserved_amount=new_reserved.to_json_value(),
                    actual_amount=capped_cost.to_json_value(),
                    discrepancy_ratio=float(discrepancy_ratio),
                    provider=provider,
                    task=task,
                    daily_balance=daily_spent.to_json_value(),
                    weekly_balance=weekly_spent.to_json_value(),
                    monthly_balance=monthly_spent.to_json_value(),
                    notes=notes,
                ),
            )

            # Also log to spend ledger for compatibility
            # Compute non-negative savings without creating negative Money [REH]
            if reserved_amount > capped_cost:
                savings = reserved_amount - capped_cost
            else:
                savings = Money.zero()

            self._append_jsonl(
                self.spend_ledger_file,
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "user_id": user_id,
                    "reserved_amount": reserved_amount.to_json_value(),
                    "actual_cost": actual_cost.to_json_value(),
                    "capped_cost": capped_cost.to_json_value(),
                    "discrepancy_ratio": float(discrepancy_ratio),
                    "savings": savings.to_json_value(),
                    "daily_total": daily_spent.to_json_value(),
                    "monthly_total": monthly_spent.to_json_value(),
                    "provider": provider,
                    "task": task,
                },
            )

            logger.info(
                f"Actual cost recorded - user: {user_id}, "
                f"actual: {actual_cost}, "
                f"capped: {capped_cost}, "
                f"reserved: {reserved_amount}, "
                f"ratio: {discrepancy_ratio:.2f}, "
                f"daily_total: {daily_spent}, "
                f"monthly_total: {monthly_spent}"
            )

    async def finalize_reservation(
        self,
        user_id: str,
        reserved_amount: Money,
        actual_cost: Money,
        *,
        job_id: Optional[str] = None,
        provider: Optional[str] = None,
        task: Optional[str] = None,
    ) -> None:
        """Alias for record_actual_cost to maintain orchestrator compatibility [REH][CA].
        This forwards to record_actual_cost with identical semantics.
        """
        await self.record_actual_cost(
            user_id=user_id,
            reserved_amount=reserved_amount,
            actual_cost=actual_cost,
            provider=provider,
            task=task,
            job_id=job_id,
        )

    async def reset_user_budget(self, user_id: str, period: str) -> None:
        """Admin command to reset user's budget [REH]"""
        async with self._get_user_lock(user_id):
            budget = await self._load_user_budget(user_id)
            if period == "daily":
                budget.set_daily_spent(Money.zero())
                budget.daily_reset_time = (
                    (datetime.now(timezone.utc) + timedelta(days=1))
                    .replace(hour=0, minute=0, second=0, microsecond=0)
                    .isoformat()
                )
            elif period == "weekly":
                budget.set_weekly_spent(Money.zero())
                budget.weekly_reset_time = (
                    (datetime.now(timezone.utc) + timedelta(days=7))
                    .replace(hour=0, minute=0, second=0, microsecond=0)
                    .isoformat()
                )
            elif period == "monthly":
                budget.set_monthly_spent(Money.zero())
                budget.monthly_reset_time = (
                    (datetime.now(timezone.utc) + timedelta(days=30))
                    .replace(hour=0, minute=0, second=0, microsecond=0)
                    .isoformat()
                )
            else:
                raise ValueError("Invalid period")

            await self._save_user_budget(budget)

            # Log transaction
            self._append_jsonl(
                self.transactions_file,
                TransactionRecord(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    user_id=user_id,
                    transaction_type="reset",
                    job_id=None,
                    amount="0.0000",
                    reserved_amount=budget.reserved_amount,
                    daily_balance=budget.daily_spent,
                    weekly_balance=budget.weekly_spent,
                    monthly_balance=budget.monthly_spent,
                    notes=f"Reset {period} budget",
                ),
            )

            logger.info(f"Reset {period} budget for user: {user_id}")

    # Convenience wrappers used by tests and admin flows
    async def reset_user_daily_budget(self, user_id: str) -> None:
        await self.reset_user_budget(user_id, "daily")

    async def reset_user_weekly_budget(self, user_id: str) -> None:
        await self.reset_user_budget(user_id, "weekly")

    async def reset_user_monthly_budget(self, user_id: str) -> None:
        await self.reset_user_budget(user_id, "monthly")

    async def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get user budget statistics [PA]"""
        async with self._get_user_lock(user_id):
            budget = await self._load_user_budget(user_id)
            self._reset_expired_periods(budget)

            daily_limit = Money(budget.daily_limit)
            daily_spent = budget.get_daily_spent()
            daily_reserved = budget.get_reserved_amount()
            daily_remaining = (
                daily_limit - daily_spent - daily_reserved
            ).clamp_minimum(0)

            weekly_limit = Money(budget.weekly_limit)
            weekly_spent = budget.get_weekly_spent()
            weekly_remaining = (
                weekly_limit - weekly_spent - daily_reserved
            ).clamp_minimum(0)

            monthly_limit = Money(budget.monthly_limit)
            monthly_spent = budget.get_monthly_spent()
            monthly_remaining = (
                monthly_limit - monthly_spent - daily_reserved
            ).clamp_minimum(0)

            return {
                "user_id": user_id,
                "daily": {
                    "limit": daily_limit.to_display_string(),
                    "spent": daily_spent.to_display_string(),
                    "reserved": daily_reserved.to_display_string(),
                    "remaining": daily_remaining.to_display_string(),
                    "reset_time": budget.daily_reset_time,
                },
                "weekly": {
                    "limit": weekly_limit.to_display_string(),
                    "spent": weekly_spent.to_display_string(),
                    "remaining": weekly_remaining.to_display_string(),
                    "reset_time": budget.weekly_reset_time,
                },
                "monthly": {
                    "limit": monthly_limit.to_display_string(),
                    "spent": monthly_spent.to_display_string(),
                    "remaining": monthly_remaining.to_display_string(),
                    "reset_time": budget.monthly_reset_time,
                },
                "total": {
                    "spent": budget.get_total_spent().to_display_string(),
                    "jobs": budget.total_jobs,
                },
                "created_at": budget.created_at,
                "updated_at": budget.updated_at,
            }

    def _format_time_until(self, target_time: datetime) -> str:
        """Format time until target as human-readable string"""
        now = datetime.now(timezone.utc)
        delta = target_time - now

        if delta.total_seconds() <= 0:
            return "now"

        hours = int(delta.total_seconds() // 3600)
        minutes = int((delta.total_seconds() % 3600) // 60)

        if hours > 24:
            days = hours // 24
            return f"{days} day{'s' if days != 1 else ''}"
        elif hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes} minute{'s' if minutes != 1 else ''}"
