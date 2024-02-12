import gym  

import math
import numpy as np
import pandas as pd

from gym import spaces

from config import Config
config = Config()

## VidyutVanika21's bidding strategy

class VV21(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4, "name": "VV21-v0"}

    def __init__(self, render_mode=None):
        self.seed_value = 0 # seed value will be overidden
        self.observation_space = spaces.Discrete(25)
        self.action_space = spaces.Box(-100.0, -10.0, shape=(1,), dtype=float)
        self.render_mode = render_mode
        self.type = "VidyutVanika21"

    def set(self, total_demand, number_of_bids, buy_limit_price_min=-100.0, buy_limit_price_max=-10.0, sell_limit_price_min=10, sell_limit_price_max=100.0, id='VV21'):
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
        self.last_uncleared_price = config.DEFAULT_MCP 
        self.action_space = spaces.Box(buy_limit_price_min, buy_limit_price_max, shape=(1,), dtype=float)
    
    def bids(self, timeslot, currentTimeslot, return_buyers_df=None, random=False, uct=False):
        rem_quantity = self.total_demand - self.cleared_demand
        
        if rem_quantity < self.min_bid_quant:
            if return_buyers_df != None:
                return_buyers_df[self.id] = None
            else:
                return None
        
        proximity = timeslot - currentTimeslot                         # Proximity runs from 24 to 1 for a day-ahead PDA
        
        bids = list()

        if rem_quantity > 0.0:

            if proximity == 24:
                return None
            
            lastaskprice = None
            
            try:
                lastaskprice = self.last_uncleared_price
            except Exception as e:
                print(e)
                lastaskprice = config.DEFAULT_MCP

            if proximity > 2:
                minprice = -max(self.sell_limit_price_min+3.0, lastaskprice-4.0)   #-0.85*lastaskprice; 
                maxprice = -(self.sell_limit_price_min+3.0);                            #-0.45*lastaskprice
                numbids = min((int)(rem_quantity/(2.0*self.min_bid_quant)), 10)

                for i in range(numbids):
                    limitprice = minprice + (i*(maxprice-minprice))/(numbids - 1 + 1e-5)
                    limitquantity = rem_quantity/numbids
                    bids.append([self.id, limitprice, limitquantity])

            elif proximity <= 2:
                laststandprice = -(lastaskprice+3.0)
                bids.append([self.id, laststandprice, rem_quantity])
        else:
            if proximity == 1:
                bids.append([self.id, config.market_order_bid_price, rem_quantity])

        if return_buyers_df != None:
            return_buyers_df[self.id] = bids
        else:
            return bids
    
    def set_cleared_demand(self, cleared_demand):
        self.cleared_demand += cleared_demand
        
    def set_last_mcp(self, mcp):
        self.last_mcp = mcp

    def set_supply(self, seller_quantities):
        self.supply = seller_quantities

    def set_demand(self, player_quantities):
        self.demand = player_quantities
    
    def set_quantities(self, player_total_demand):
        self.quantities = player_total_demand

    def update_buy_limit_price_max(self, price):
        # self.buy_limit_price_max = max(self.buy_limit_price_min, price)
        pass  # do not narrow the limitprice range, miso is selling so need to place lower limitprices

    def update_last_uncleared_price(self, price):
        self.last_uncleared_price = price

    def reset(self):
        pass

    def step(self):
        pass

gym.envs.register(
    id='VV21-v0',
    entry_point='broker.vv21:VV21',
)