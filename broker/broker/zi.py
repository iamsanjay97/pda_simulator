import gym  

import math
import numpy as np
import pandas as pd

from gym import spaces

from config import Config
config = Config()

## A Randomized Way to Generate Buyer's Bids

class ZI(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4, "name": "ZI-v0"}

    def __init__(self, render_mode=None):
        self.seed_value = 0 # seed value will be overidden

        self.observation_space = spaces.Box(0, 1, shape=(2,), dtype=float)
        self.action_space = spaces.Discrete(4)
        self.render_mode = render_mode

    def set(self, total_demand, number_of_bids, buy_limit_price_min=-100.0, buy_limit_price_max=-1.0, sell_limit_price_min=0.5, sell_limit_price_max=100.0, id='ZI'):
        self.id = id
        self.total_demand = total_demand 
        self.cleared_demand = 0
        self.min_bid_quant = 0.01
        self.number_of_bids = number_of_bids
        self.buy_limit_price_min = buy_limit_price_min
        self.buy_limit_price_max = buy_limit_price_max
        self.sell_limit_price_min = sell_limit_price_min
        self.sell_limit_price_max = sell_limit_price_max
        self.last_mcp = config.DEFAULT_MCP
 
    def gen_function(self, rem_quantity):
        price = np.random.uniform(self.buy_limit_price_min, self.buy_limit_price_max)
        quantity = max(rem_quantity/self.number_of_bids, self.min_bid_quant)
        return price, quantity
    
    def bids(self, timeslot, currentTimeslot, random=False):
        rem_quantity = self.total_demand - self.cleared_demand
        
        if rem_quantity < self.min_bid_quant:
            return None
            
        bids = list()
        for i in range(self.number_of_bids):
            price, quantity = self.gen_function(rem_quantity)
            bids.append([self.id, price, quantity])

        return bids
    
    def set_cleared_demand(self, cleared_demand):
        self.cleared_demand += cleared_demand
        
    def set_last_mcp(self, mcp):
        self.last_mcp = mcp

    def reset(self):
        pass

    def step(self):
        pass

gym.envs.register(
    id='ZI-v0',
    entry_point='broker.zi:ZI',
)