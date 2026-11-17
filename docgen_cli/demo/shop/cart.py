"""A minimal shopping cart."""

from .pricing import add_tax, apply_discount


class Cart:
    def __init__(self, discount_percent=0):
        self.items = {}
        self.discount_percent = discount_percent

    def add_item(self, name, price, quantity=1):
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        entry = self.items.setdefault(name, {"price": price, "quantity": 0})
        entry["quantity"] += quantity

    def remove_item(self, name):
        if name not in self.items:
            raise KeyError(name)
        del self.items[name]

    def subtotal(self):
        return sum(e["price"] * e["quantity"] for e in self.items.values())

    def total(self):
        discounted = apply_discount(self.subtotal(), self.discount_percent)
        return add_tax(discounted)

    @property
    def item_count(self):
        return sum(e["quantity"] for e in self.items.values())

    def __len__(self):
        return len(self.items)

    def clear(self):
        """Remove every item from the cart."""
        self.items.clear()
