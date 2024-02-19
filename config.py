## Configuration file having all the configurable parameters

import numpy as np

class Config:

    def __init__(self):
        self.k = 0.5                                    # factor k for k-double auction clearning
        self.default_margin = 0.05                      # margin when seller's ask is market-order
        self.market_demand = np.random.normal(200, 10)  # randomly generated market demand (in mwh)
        self.market_order_bid_price = 1e9
        self.market_order_ask_price = 0
        self.DEFAULT_MCP = 80
        self.HOUR_AHEAD_AUCTIONS = 24
        self.NUMBER_OF_ROLLOUTS = 50
        self.balancing_price = 90.0
        self.parallel = False                           # to place bids parallaly
        self.iters = 2                                # number of iterations for comparison

        self.proximity_record_min = 0              
        self.proximity_record_max = 24
        self.balancing_price_record_min = 0            
        self.balancing_price_record_max = 200.0         # self.balancing_price
        self.quantity_record_min = 0              
        self.quantity_record_max = self.market_demand

# Demand Levels: Low, Medium, High, Extreme
# MWh          : 100, 200, 300, 400