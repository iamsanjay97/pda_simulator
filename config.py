## Configuration file having all the configurable parameters

import numpy as np

class Config:

    def __init__(self):
        self.k = 0.5                                    # factor k for k-double auction clearning
        self.default_margin = 0.05                      # margin when seller's ask is market-order
        self.market_demand = np.random.normal(120, 10)  # randomly generated market demand (in mwh)
        self.market_order_bid_price = 1e9
        self.market_order_ask_price = 0
        self.DEFAULT_MCP = 40
        self.HOUR_AHEAD_AUCTIONS = 24
        self.NUMBER_OF_ROLLOUTS = 1000