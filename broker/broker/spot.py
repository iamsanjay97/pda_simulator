import sys
import gym  

import math
import numpy as np
import pandas as pd

from gym import spaces
from config import Config

sys.path.append('/mnt/d/PowerTAC/TCS/mcts_thread/pda_simulator/broker/broker/spot')
from mcts import MCTS, Action, ACTION_TYPE, Observer

config = Config()

## SPOT's bidding strategy

class SPOT(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4, "name": "SPOT-v0"}

    server_min_mwh = 0.01
    min_limit_mwh = 2.0
    start_timeslot = 360
    wholesale_current_timeslot = 360

    first_time_flags = True

    def __init__(self, render_mode=None):
        self.seed_value = 0 # seed value will be overidden
        self.observation_space = spaces.Discrete(25)
        self.action_space = spaces.Box(-100.0, -10.0, shape=(1,), dtype=float)
        self.render_mode = render_mode
        self.type = "SPOT"
        self.balancing_price = 200.0
        self.mcts = MCTS()
        self.observer = Observer()

    def version(self):
        return "SPOT's Bidding Strategy"
    
    def action(self, currentTimeslot,):
         self.wholesaleCurrentTimeslot = currentTimeslot
    
    def set(self, total_demand, number_of_bids=1, buy_limit_price_min=-100.0, buy_limit_price_max=-1.0, sell_limit_price_min=0.5, sell_limit_price_max=90.0, id='SPOT'):
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
    
    def bids(self, timeslot, current_timeslot, return_buyers_df=None, random=False):

        rem_quantity = self.total_demand - self.cleared_demand
        
        if rem_quantity < self.min_bid_quant:
            if return_buyers_df != None:
                return_buyers_df[self.id] = None
            else:
                return None

        proximity = timeslot - current_timeslot

        try:
            bids = list()
            
            days = (current_timeslot - self.start_timeslot) / 24
            hour = (current_timeslot - self.start_timeslot) % 24
                        
            if self.first_time_flags == True:
                self.mcts.setup()
                self.first_time_flags = False

            self.observer.set_time(days, hour, proximity, current_timeslot)
            
            self.observer.needed_enery_per_broker = rem_quantity
            self.observer.initial_needed_enery_mcts_broker = rem_quantity
            
            if abs(rem_quantity) <= self.server_min_mwh:
                return

            # TO DO: Check what to do for the predictor
            try:
                market_limit_price = self.market_clearing_price_prediction.get(days)
            except:
                market_limit_price = None
                     
            for haid in range(24):
                predicted_clearing_price_per_auction = 30.0
                
                if market_limit_price != None:                
                    if (market_limit_price.predicted_clearing_price_per_auction[hour][haid] != 0.0):
                        predicted_clearing_price_per_auction = market_limit_price.predicted_clearing_price_per_auction[hour][haid]
                else:
                    predicted_clearing_price_per_auction = 100.0   # Fixed value as no price predictor
                self.observer.arr_clearing_prices[haid] = predicted_clearing_price_per_auction

            best_move = self.mcts.get_best_mcts_move(self.observer)

            number_of_bids = 10
            unit_price_increment = 1
            limit_price = abs(best_move.minmcts_clearing_price)
            price_range = abs(best_move.maxmcts_clearing_price - best_move.minmcts_clearing_price)
            
            min_mwh = rem_quantity / number_of_bids
            if abs(min_mwh) <= self.server_min_mwh:
                min_mwh = rem_quantity
                number_of_bids = 1 
            
            unit_price_increment = price_range / number_of_bids
            
            if best_move.nobid != False:               
                if best_move.actionType == ACTION_TYPE.BUY:
                    if rem_quantity > 0.0:
                        surplus = abs(rem_quantity) * (best_move.vol_percentage - 1.0)
                        totalE = surplus + abs(rem_quantity)
                        min_mwh = totalE / number_of_bids
                        
                        if limit_price > 0:
                            limit_price *= -1.0
                            unit_price_increment *= -1.0
                        
                        for i in range (number_of_bids+1):
                            bids.append([self.id, bprice, min_mwh])
                            limit_price += unit_price_increment
                else:
                    surplus = -rem_quantity
                    min_mwh = abs(rem_quantity) * (best_move.vol_percentage - 1)
                    if min_mwh < surplus:
                        min_mwh = surplus
        
                    bprice = self.balancing_price
                    
                    min_mwh /= number_of_bids
                    
                    if min_mwh > 0.0:
                        min_mwh *= -1.0
                    
                    if bprice < 0.0:
                        bprice *= -1.0
                    
                    if unit_price_increment < 0.0:
                        unit_price_increment *= -1.0
            
                    for i in range (number_of_bids+1):
                        bids.append([self.id, bprice, min_mwh])
                        bprice += unit_price_increment

            if return_buyers_df != None:
                return_buyers_df[self.id] = bids
            else:
                return bids

        except Exception as E:
            print(E)
    
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
    id='SPOT-v0',
    entry_point='broker.spot:SPOT',
)