"""Pricing helpers for the demo shop."""


def apply_discount(price, percent):
    if percent < 0 or percent > 100:
        raise ValueError("percent must be between 0 and 100")
    return round(price * (1 - percent / 100), 2)


def add_tax(price, rate=0.07):
    return round(price * (1 + rate), 2)


def split_bill(total_cents, people):
    base, extra = divmod(total_cents, people)
    return [base + 1 if i < extra else base for i in range(people)]


def bulk_price(unit_price, quantity):
    """Return the price for a quantity, with 10% off for 10 or more."""
    total = unit_price * quantity
    return round(total * 0.9, 2) if quantity >= 10 else round(total, 2)


def cheapest(items):
    best = None
    for item in items:
        if best is None or item["price"] < best["price"]:
            best = item
    return best
