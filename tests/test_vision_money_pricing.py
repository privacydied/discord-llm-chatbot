#!/usr/bin/env python3
"""
Test suite for vision Money type and pricing system [REH][PA]

Tests:
- Money arithmetic and precision
- Pricing table loading and estimation
- Provider usage normalization
- Budget manager with atomic writes
- Discrepancy detection and capping
"""

import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from decimal import Decimal
from datetime import datetime, timezone

from bot.vision.money import Money
from bot.vision.pricing_loader import PricingTable, get_pricing_table
from bot.vision.budget_manager_v2 import VisionBudgetManager, UserBudget, BudgetResult
from bot.vision.types import VisionTask, VisionProvider, VisionRequest


class TestMoney:
    """Test Money type for precision and safety [REH]"""
    
    def test_money_creation(self):
        """Test Money creation from various types"""
        # From float
        m1 = Money(10.50)
        assert m1.to_float() == 10.50
        assert m1.to_display_string() == "$10.50"
        
        # From string
        m2 = Money("10.50")
        assert m2 == m1
        
        # From int
        m3 = Money(10)
        assert m3.to_display_string() == "$10.00"
        
        # From Decimal
        m4 = Money(Decimal("10.5000"))
        assert m4 == m1
        
        # From another Money
        m5 = Money(m1)
        assert m5 == m1
    
    def test_money_precision(self):
        """Test Money maintains precision"""
        # Test internal precision (4 decimal places)
        m1 = Money("0.0001")
        assert m1.to_json_value() == "0.0001"
        
        # Test rounding
        m2 = Money("0.00005")  # Should round to 0.0001
        assert m2.to_json_value() == "0.0001"
        
        # Test display precision (2 decimal places)
        m3 = Money("10.999")
        assert m3.to_display_string() == "$11.00"
        assert m3.to_display_string(precision=3) == "$10.999"
    
    def test_money_arithmetic(self):
        """Test Money arithmetic operations"""
        m1 = Money("10.00")
        m2 = Money("5.50")
        
        # Addition
        m3 = m1 + m2
        assert m3.to_float() == 15.50
        assert isinstance(m3, Money)
        
        # Subtraction
        m4 = m1 - m2
        assert m4.to_float() == 4.50
        
        # Multiplication by scalar
        m5 = m1 * 2
        assert m5.to_float() == 20.00
        
        # Division by scalar
        m6 = m1 / 2
        assert m6.to_float() == 5.00
        
        # Chain operations
        m7 = (m1 + m2) * 1.1
        assert m7.to_float() == pytest.approx(17.05, rel=1e-4)
    
    def test_money_comparison(self):
        """Test Money comparison operations"""
        m1 = Money("10.00")
        m2 = Money("10.00")
        m3 = Money("5.00")
        m4 = Money("15.00")
        
        assert m1 == m2
        assert m1 != m3
        assert m3 < m1
        assert m1 > m3
        assert m3 <= m1
        assert m1 >= m3
        assert m1 <= m4
    
    def test_money_utilities(self):
        """Test Money utility methods"""
        # Zero
        m1 = Money.zero()
        assert m1.is_zero()
        assert not m1.is_positive()
        
        # Positive
        m2 = Money("10.00")
        assert not m2.is_zero()
        assert m2.is_positive()
        
        # From cents
        m3 = Money.from_cents(1050)
        assert m3.to_float() == 10.50
        
        # From credits
        m4 = Money.from_credits(100, credits_per_dollar=100)
        assert m4.to_float() == 1.00
        
        # Clamp minimum
        m5 = Money("-5.00")
        m6 = m5.clamp_minimum(0)
        assert m6.to_float() == 0.00
        
        # Ratio
        m7 = Money("20.00")
        m8 = Money("10.00")
        ratio = m7.ratio_to(m8)
        assert float(ratio) == 2.0


class TestPricingTable:
    """Test pricing table and estimation [PA]"""
    
    @pytest.fixture
    def pricing_table(self, tmp_path):
        """Create test pricing table"""
        pricing_data = {
            "version": "test",
            "currency": "USD",
            "providers": {
                "novita": {
                    "base_prices": {
                        "text_to_image": {
                            "base_cost": 0.02,
                            "per_image": 0.02,
                            "size_multipliers": {
                                "512x512": 1.0,
                                "1024x1024": 1.5,
                                "2048x2048": 3.0
                            },
                            "model_multipliers": {
                                "default": 1.0,
                                "flux": 1.5
                            }
                        },
                        "text_to_video": {
                            "base_cost": 0.10,
                            "per_second": 0.05
                        }
                    },
                    "credits_per_dollar": 100,
                    "safety_factor": 1.1
                }
            },
            "default_estimates": {
                "text_to_image": 0.02,
                "text_to_video": 0.30
            },
            "estimation_config": {
                "safety_factor": 1.2,
                "max_discrepancy_ratio": 5.0,
                "warning_discrepancy_ratio": 2.0
            }
        }
        
        pricing_file = tmp_path / "test_pricing.json"
        with open(pricing_file, 'w') as f:
            json.dump(pricing_data, f)
        
        return PricingTable(pricing_file)
    
    def test_estimate_image_cost(self, pricing_table):
        """Test image generation cost estimation"""
        # Basic 1024x1024 image
        cost = pricing_table.estimate_cost(
            provider=VisionProvider.NOVITA,
            task=VisionTask.TEXT_TO_IMAGE,
            width=1024,
            height=1024,
            num_images=1
        )
        # 0.02 * 1.5 (size) * 1.1 (provider) * 1.2 (global) = 0.0396
        assert cost.to_float() == pytest.approx(0.0396, rel=1e-3)
        
        # Multiple images
        cost2 = pricing_table.estimate_cost(
            provider=VisionProvider.NOVITA,
            task=VisionTask.TEXT_TO_IMAGE,
            width=512,
            height=512,
            num_images=3
        )
        # 0.02 * 3 * 1.0 (size) * 1.1 * 1.2 = 0.0792
        assert cost2.to_float() == pytest.approx(0.0792, rel=1e-3)
        
        # With model multiplier
        cost3 = pricing_table.estimate_cost(
            provider=VisionProvider.NOVITA,
            task=VisionTask.TEXT_TO_IMAGE,
            width=1024,
            height=1024,
            num_images=1,
            model="flux"  # Use model name that exists in test pricing
        )
        # 0.02 * 1.5 (size) * 1.5 (model) * 1.1 * 1.2 = 0.0594
        assert cost3.to_float() == pytest.approx(0.0594, rel=1e-3)
    
    def test_estimate_video_cost(self, pricing_table):
        """Test video generation cost estimation"""
        cost = pricing_table.estimate_cost(
            provider=VisionProvider.NOVITA,
            task=VisionTask.TEXT_TO_VIDEO,
            duration_seconds=4.0
        )
        # (0.10 + 0.05 * 4) * 1.1 * 1.2 = 0.396
        assert cost.to_float() == pytest.approx(0.396, rel=1e-3)
    
    def test_normalize_provider_usage(self, pricing_table):
        """Test provider usage normalization"""
        # Credits-based
        cost1 = pricing_table.normalize_provider_usage(
            provider=VisionProvider.NOVITA,
            usage_data={"credits": 200}
        )
        assert cost1.to_float() == 2.00  # 200 credits / 100 per dollar
        
        # Direct USD
        cost2 = pricing_table.normalize_provider_usage(
            provider=VisionProvider.NOVITA,
            usage_data={"cost_usd": 1.50}
        )
        assert cost2.to_float() == 1.50
        
        # Cents
        cost3 = pricing_table.normalize_provider_usage(
            provider=VisionProvider.NOVITA,
            usage_data={"cents": 150}
        )
        assert cost3.to_float() == 1.50


class TestBudgetManager:
    """Test budget manager with Money type [REH][RM]"""
    
    @pytest.fixture
    async def budget_manager(self, tmp_path):
        """Create test budget manager"""
        config = {
            "VISION_DATA_DIR": str(tmp_path),
            "VISION_DAILY_LIMIT": 5.0,
            "VISION_WEEKLY_LIMIT": 20.0,
            "VISION_MONTHLY_LIMIT": 50.0
        }
        return VisionBudgetManager(config)
    
    @pytest.mark.asyncio
    async def test_budget_check(self, budget_manager):
        """Test budget checking with reservations"""
        request = VisionRequest(
            user_id="test_user",
            task=VisionTask.TEXT_TO_IMAGE,
            prompt="test",
            estimated_cost=0.50
        )
        
        # First check should pass
        result = await budget_manager.check_budget(request)
        assert result.approved
        assert result.remaining_budget >= Money("4.50")
        
        # Reserve some budget
        await budget_manager.reserve_budget("test_user", Money("2.00"))
        
        # Check with reservation considered
        result2 = await budget_manager.check_budget(request)
        assert result2.approved
        assert result2.remaining_budget >= Money("2.50")
        
        # Large request should fail
        request.estimated_cost = 4.00
        result3 = await budget_manager.check_budget(request)
        assert not result3.approved
        assert "Daily budget limit reached" in result3.user_message
    
    @pytest.mark.asyncio
    async def test_record_actual_cost(self, budget_manager):
        """Test recording actual cost with discrepancy handling"""
        user_id = "test_user"
        
        # Reserve budget
        reserved = Money("1.00")
        await budget_manager.reserve_budget(user_id, reserved)
        
        # Record actual cost (within normal range)
        actual = Money("0.90")
        await budget_manager.record_actual_cost(
            user_id=user_id,
            reserved_amount=reserved,
            actual_cost=actual,
            provider="novita",
            task="text_to_image"
        )
        
        # Check budget was updated correctly
        stats = await budget_manager.get_user_stats(user_id)
        assert stats["daily"]["spent"] == "$0.90"
        assert stats["daily"]["reserved"] == "$0.00"
        
        # Test discrepancy capping (6x estimate)
        reserved2 = Money("1.00")
        await budget_manager.reserve_budget(user_id, reserved2)
        
        actual2 = Money("6.00")  # 6x the estimate
        await budget_manager.record_actual_cost(
            user_id=user_id,
            reserved_amount=reserved2,
            actual_cost=actual2,
            provider="novita",
            task="text_to_image"
        )
        
        # Should be capped at 5x
        stats2 = await budget_manager.get_user_stats(user_id)
        daily_spent = Money(stats2["daily"]["spent"].replace("$", ""))
        assert daily_spent.to_float() == pytest.approx(5.90, rel=1e-3)  # 0.90 + 5.00
    
    @pytest.mark.asyncio
    async def test_atomic_writes(self, budget_manager, tmp_path):
        """Test atomic file writes"""
        user_id = "test_user"
        
        # Create initial budget
        await budget_manager.reserve_budget(user_id, Money("1.00"))
        
        # Verify files exist
        budgets_file = tmp_path / "budgets.json"
        transactions_file = tmp_path / "transactions.jsonl"
        
        assert budgets_file.exists()
        assert transactions_file.exists()
        
        # Verify JSON is valid
        with open(budgets_file, 'r') as f:
            data = json.load(f)
            assert user_id in data
            assert data[user_id]["reserved_amount"] == "1.0000"
        
        # Verify transaction log
        with open(transactions_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) >= 1
            # Parse the transaction log entry
            for line in lines:
                if line.strip():
                    transaction = json.loads(line)
                    if transaction.get("transaction_type") == "reserve":
                        assert transaction["amount"] == "1.0000"
                        break
    
    @pytest.mark.asyncio
    async def test_period_resets(self, budget_manager):
        """Test budget period resets"""
        user_id = "test_user"
        
        # Record some spend
        await budget_manager.reserve_budget(user_id, Money("2.00"))
        await budget_manager.record_actual_cost(
            user_id=user_id,
            reserved_amount=Money("2.00"),
            actual_cost=Money("1.80")
        )
        
        # Check current stats
        stats = await budget_manager.get_user_stats(user_id)
        assert stats["daily"]["spent"] == "$1.80"
        
        # Manually reset daily budget
        await budget_manager.reset_user_daily_budget(user_id)
        
        # Check reset worked
        stats2 = await budget_manager.get_user_stats(user_id)
        assert stats2["daily"]["spent"] == "$0.00"
        assert stats2["weekly"]["spent"] == "$1.80"  # Weekly not reset
        assert stats2["monthly"]["spent"] == "$1.80"  # Monthly not reset


class TestIntegration:
    """Integration tests for the full system [REH]"""
    
    @pytest.mark.asyncio
    async def test_full_flow(self, tmp_path):
        """Test complete flow from estimation to recording"""
        # Setup
        config = {
            "VISION_DATA_DIR": str(tmp_path),
            "VISION_DAILY_LIMIT": 5.0,
            "VISION_WEEKLY_LIMIT": 20.0,
            "VISION_MONTHLY_LIMIT": 50.0
        }
        
        budget_manager = VisionBudgetManager(config)
        pricing_table = get_pricing_table()
        
        # Create request
        request = VisionRequest(
            user_id="test_user",
            task=VisionTask.TEXT_TO_IMAGE,
            prompt="test image",
            width=1024,
            height=1024,
            batch_size=1  # Use batch_size instead of num_images
        )
        
        # Estimate cost
        estimated = pricing_table.estimate_cost(
            provider=VisionProvider.NOVITA,
            task=request.task,
            width=request.width,
            height=request.height,
            num_images=request.batch_size
        )
        request.estimated_cost = estimated.to_float()
        
        # Check budget
        result = await budget_manager.check_budget(request)
        assert result.approved
        
        # Reserve budget
        await budget_manager.reserve_budget(request.user_id, estimated)
        
        # Simulate provider response with actual cost
        actual_usage = {"credits": 2.5}  # 2.5 credits = $0.025
        actual_cost = pricing_table.normalize_provider_usage(
            provider=VisionProvider.NOVITA,
            usage_data=actual_usage
        )
        
        # Record actual cost
        await budget_manager.record_actual_cost(
            user_id=request.user_id,
            reserved_amount=estimated,
            actual_cost=actual_cost,
            provider="novita",
            task="text_to_image"
        )
        
        # Verify final state
        stats = await budget_manager.get_user_stats(request.user_id)
        assert stats["daily"]["reserved"] == "$0.00"
        assert float(stats["daily"]["spent"].replace("$", "")) > 0
        assert stats["total"]["jobs"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
