from bot.vision.money import Money


def test_money_supports_float_coercion() -> None:
    amount = Money("1.2345")
    assert float(amount) == amount.to_float()
