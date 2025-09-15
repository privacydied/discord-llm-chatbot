"""
Money type for precise financial calculations [CA][CMV]

Provides a type-safe Money wrapper around Decimal for all financial operations.
All monetary values are stored and calculated in USD with 4 decimal places internally,
displayed with 2-3 decimal places to users.

Follows Clean Architecture (CA) and Constants over Magic Values (CMV) principles.
"""

from __future__ import annotations
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Union

from bot.utils.logging import get_logger

logger = get_logger(__name__)


class Money:
    """
    Type-safe money representation using Decimal for precise arithmetic.

    All values are in USD. Internal precision is 4 decimal places,
    display precision is 2-3 decimal places.
    """

    # Constants [CMV]
    CURRENCY = "USD"
    INTERNAL_PRECISION = Decimal("0.0001")  # 4 decimal places
    DISPLAY_PRECISION = Decimal("0.01")  # 2 decimal places for users
    ZERO = Decimal("0")

    def __init__(self, value: Union[str, int, float, Decimal, "Money"]) -> None:
        """
        Initialize Money from various types.

        Args:
            value: Amount in USD (str, int, float, Decimal, or Money)

        Raises:
            ValueError: If value cannot be converted to valid money amount
        """
        if isinstance(value, Money):
            self._amount = value._amount
        elif isinstance(value, Decimal):
            self._amount = value.quantize(
                self.INTERNAL_PRECISION, rounding=ROUND_HALF_UP
            )
        else:
            try:
                # Convert to Decimal first, then quantize
                decimal_value = Decimal(str(value))
                self._amount = decimal_value.quantize(
                    self.INTERNAL_PRECISION, rounding=ROUND_HALF_UP
                )
            except (InvalidOperation, ValueError) as e:
                logger.error(f"Invalid money value: {value} - {e}")
                raise ValueError(f"Cannot convert {value} to Money: {e}")

        # Ensure non-negative for costs
        if self._amount < self.ZERO:
            logger.warning(f"Negative money value created: {self._amount}")

    @classmethod
    def from_cents(cls, cents: Union[int, float]) -> Money:
        """Create Money from cent value (useful for provider APIs that return cents)"""
        return cls(Decimal(str(cents)) / 100)

    @classmethod
    def from_credits(
        cls, credits: Union[int, float], credits_per_dollar: float = 100
    ) -> Money:
        """Create Money from provider credits with configurable exchange rate"""
        return cls(Decimal(str(credits)) / Decimal(str(credits_per_dollar)))

    @classmethod
    def zero(cls) -> Money:
        """Return zero money value"""
        return cls(cls.ZERO)

    def to_decimal(self) -> Decimal:
        """Get raw Decimal value"""
        return self._amount

    def to_float(self) -> float:
        """Get float value (for legacy compatibility only)"""
        return float(self._amount)

    def to_display_string(self, precision: int = 2) -> str:
        """Format for user display with $ symbol"""
        if precision == 2:
            quantized = self._amount.quantize(
                self.DISPLAY_PRECISION, rounding=ROUND_HALF_UP
            )
        else:
            precision_str = f"0.{'0' * precision}"
            quantized = self._amount.quantize(
                Decimal(precision_str), rounding=ROUND_HALF_UP
            )
        return f"${quantized}"

    def to_json_value(self) -> str:
        """Serialize to JSON-safe string value"""
        return str(self._amount)

    @classmethod
    def from_json_value(cls, value: str) -> Money:
        """Deserialize from JSON string value"""
        return cls(value)

    # Arithmetic operations
    def __add__(self, other: Union[Money, Decimal, int, float]) -> Money:
        if not isinstance(other, Money):
            other = Money(other)
        return Money(self._amount + other._amount)

    def __sub__(self, other: Union[Money, Decimal, int, float]) -> Money:
        if not isinstance(other, Money):
            other = Money(other)
        return Money(self._amount - other._amount)

    def __mul__(self, factor: Union[int, float, Decimal]) -> Money:
        """Multiply money by a scalar factor"""
        return Money(self._amount * Decimal(str(factor)))

    def __truediv__(self, divisor: Union[int, float, Decimal]) -> Money:
        """Divide money by a scalar divisor"""
        return Money(self._amount / Decimal(str(divisor)))

    # Comparison operations
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, (Money, int, float, Decimal)):
            return False
        if not isinstance(other, Money):
            other = Money(other)
        return self._amount == other._amount

    def __lt__(self, other: Union[Money, Decimal, int, float]) -> bool:
        if not isinstance(other, Money):
            other = Money(other)
        return self._amount < other._amount

    def __le__(self, other: Union[Money, Decimal, int, float]) -> bool:
        if not isinstance(other, Money):
            other = Money(other)
        return self._amount <= other._amount

    def __gt__(self, other: Union[Money, Decimal, int, float]) -> bool:
        if not isinstance(other, Money):
            other = Money(other)
        return self._amount > other._amount

    def __ge__(self, other: Union[Money, Decimal, int, float]) -> bool:
        if not isinstance(other, Money):
            other = Money(other)
        return self._amount >= other._amount

    def __str__(self) -> str:
        """String representation for logging"""
        return f"{self._amount} {self.CURRENCY}"

    def __repr__(self) -> str:
        """Developer representation"""
        return f"Money('{self._amount}')"

    def __hash__(self) -> int:
        """Make Money hashable for use in sets/dicts"""
        return hash((self._amount, self.CURRENCY))

    # Utility methods
    def clamp_minimum(self, minimum: Union[Money, Decimal, int, float] = 0) -> Money:
        """Ensure money is at least the minimum value (default 0)"""
        if not isinstance(minimum, Money):
            minimum = Money(minimum)
        if self._amount < minimum._amount:
            return Money(minimum._amount)
        return self

    def is_zero(self) -> bool:
        """Check if amount is zero"""
        return self._amount == self.ZERO

    def is_positive(self) -> bool:
        """Check if amount is positive (> 0)"""
        return self._amount > self.ZERO

    def ratio_to(self, other: Union[Money, Decimal, int, float]) -> Decimal:
        """Calculate ratio of this money to another (for discrepancy checks)"""
        if not isinstance(other, Money):
            other = Money(other)
        if other._amount == self.ZERO:
            return Decimal("0") if self._amount == self.ZERO else Decimal("999999")
        return self._amount / other._amount
