import sys
import gym  

import math
import numpy as np
import pandas as pd

from gym import spaces

# sys.path.append('/home/sanjay/Research/MCTS/Codes/pda_simulator/broker/broker')
from config import Config

## A Randomized Way to Generate Buyer's Bids

config = Config()

class LimitPriceDeterminants:
    
    def __init__(self, limitPrice, profitMargin, executionTimeslot):
    
        self.limitPrice = limitPrice
        self.profitMargin = profitMargin
        self.delta = 0.0
        self.executionTimeslot = executionTimeslot

        self.learningRate = 0.8
        self.momentumCoefficient = 0.5

class ZIP(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4, "name": "ZIP-v0"}

    def __init__(self, render_mode=None):
        self.seed_value = 0 # seed value will be overidden

        self.observation_space = spaces.Box(0, 1, shape=(2,), dtype=float)
        self.action_space = spaces.Discrete(4)
        self.render_mode = render_mode
        self.type = "ZIP"

    def set(self, total_demand, number_of_bids=1, buy_limit_price_min=-100.0, buy_limit_price_max=-1.0, sell_limit_price_min=0.5, sell_limit_price_max=100.0, id='ZIP'):
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

        self.bid_price_by_future_timeslot = dict()
        self.ask_price_by_future_timeslot = dict()


    def set_cleared_demand(self, cleared_demand):
        self.cleared_demand += cleared_demand
        
    def set_last_mcp(self, mcp):
        self.last_mcp = mcp

    def bids(self, timeslot, currentTimeslot, random=False, uct=False):

        rem_quantity = self.total_demand - self.cleared_demand
        
        if rem_quantity < self.min_bid_quant:
            return None
        
        amount_needed = rem_quantity
        remaining_tries = timeslot - currentTimeslot
        
        bids = list()
        if remaining_tries > 0:
            
            if amount_needed > 0.0:
                
                if timeslot not in self.bid_price_by_future_timeslot.keys():
                    lpd = LimitPriceDeterminants(np.random.random()*self.buy_limit_price_min, -0.1 , currentTimeslot)
                    self.bid_price_by_future_timeslot.update({timeslot: lpd})
                    bid = lpd.limitPrice
                else:
                    lpd = self.bid_price_by_future_timeslot.get(timeslot)
                    sdelta = lpd.learningRate * (-self.last_mcp - lpd.limitPrice)         # For bid, limitPrice is negetive, so "-lastMCP"
                    lpd.executionTimeslot = currentTimeslot
                    lpd.delta = lpd.momentumCoefficient * lpd.delta + (1 - lpd.momentumCoefficient) * sdelta
                    lpd.profitMargin = max(-1.0, lpd.profitMargin + (lpd.delta/lpd.limitPrice))
                    lpd.limitPrice = max(self.buy_limit_price_min, lpd.limitPrice * (1 + lpd.profitMargin))
                    self.bid_price_by_future_timeslot.update({timeslot: lpd})
                    bid = lpd.limitPrice
                
            else:
                
                if timeslot not in self.ask_price_by_future_timeslot.keys():
                    lpd = LimitPriceDeterminants((config.sell_limit_price_max + config.sell_limit_price_min)/2, 0.1, currentTimeslot)
                    self.ask_price_by_future_timeslot.update({timeslot: lpd})
                    bid = lpd.limitPrice
                else:
                    lpd = self.ask_price_by_future_timeslot.get(timeslot)
                    sdelta = lpd.learningRate * (self.last_mcp - lpd.limitPrice) 
                    lpd.executionTimeslot = currentTimeslot
                    lpd.delta = lpd.momentumCoefficient * lpd.delta + (1 - lpd.momentumCoefficient) * sdelta
                    lpd.profitMargin = max(-1.0, lpd.profitMargin + (lpd.delta/lpd.limitPrice))
                    lpd.limitPrice = max(self.sell_limit_price_max, lpd.limitPrice * (1 + lpd.profitMargin))
                    self.ask_price_by_future_timeslot.update({timeslot: lpd})
                    bid = lpd.limitPrice

            bids.append([self.id, bid, amount_needed])
        else:
            bids.append([self.id, -config.market_order_bid_price, amount_needed])

        return bids
    
    def reset(self):
        pass

    def step(self):
        pass


gym.envs.register(
    id='ZIP-v0',
    entry_point='broker.zip:ZIP',
)