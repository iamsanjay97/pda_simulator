class Action(object):
    """Represents an action with adjustable price calculation."""

    def __init__(self, action_name: str, min_mult: float, max_mult: float, no_bid: bool, action_type: str, percentage: float):
        """Initializes the Action object."""
        self.action_name = action_name
        self.min_mult = min_mult
        self.max_mult = max_mult
        self.no_bid = no_bid
        self.type = action_type
        self.percentage = percentage

    def get_adjusted_price(self, mean_price: float, stddev: float) -> tuple[float, float]:
        """Calculates and returns the adjusted price range."""
        min_price = mean_price + self.min_mult * stddev
        max_price = mean_price + self.max_mult * stddev
        if min_price > max_price:
            min_price, max_price = max_price, min_price
        return min_price, max_price

    def __str__(self) -> str:
        """Returns a string representation of the Action object."""
        return f"[action: {self.action_name} minMult: {self.min_mult} maxMult: {self.max_mult} nobid: {self.no_bid}]"


class ACTION_TYPE:
    """Enum for action types."""

    BUY = "BUY"
    SELL = "SELL"
    NO_BID = "NO_BID"
