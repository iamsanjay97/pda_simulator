import sys
import gym
import numpy as np
import pandas as pd

from gym import spaces
from collections import OrderedDict
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# sys.path.append('/home/sanjay/Research/MCTS/Codes/pda_simulator/pdauction/pdauction/envs')
from config import Config

config = Config()

class AuctioneerEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4, "name": "Auctioneer-v0"}

    def __init__(self, render_mode=None):

        self.observation_space = spaces.Box(0, 1, shape=(2,), dtype=float)
        self.action_space = spaces.Discrete(4)
        self.render_mode = render_mode


    def plot(self, asks_df, bids_df):
        ask_prices = list(asks_df['Price'])
        ask_quantities = list(asks_df['Quantity'])
        
        ask_prices.insert(0, ask_prices[0])
        ask_quantities.insert(0, 0)
        ask_cum_quantities = np.cumsum(ask_quantities)
        
        bid_prices = list(bids_df['Price'])
        bid_quantities = list(bids_df['Quantity'])
        
        bid_prices.insert(0, bid_prices[0])
        bid_quantities.insert(0, 0)
        bid_cum_quantities = np.cumsum(bid_quantities)
                
        plt.step(np.negative(ask_cum_quantities), ask_prices, where='pre')
        plt.step(bid_cum_quantities, np.negative(bid_prices), where='pre')
        plt.show()


    def clearing_mechanism(self, asks_df, bids_df):
       
        total_mwh = 0.0
        mcp = 40.0                                    # default macp when both ask and bid are market order                      
        cleared_asks = list()
        cleared_bids = list()
        last_uncleared_ask = asks_df[:1].values[0][1] 

        i = 0

        while (not asks_df.empty and not bids_df.empty and (-bids_df[:1].values[0][1] > asks_df[:1].values[0][1])):
            i += 1

            bid = bids_df[:1].values[0]               # a single bid in the form of ['ID', 'Price', 'Quantity']
            ask = asks_df[:1].values[0]               # a single ask in the form of ['ID', 'Price', 'Quantity']

            transfer = min(bid[2], -ask[2])           # index 2 is for Quantity

            total_mwh += transfer
            if (-bid[1] != config.market_order_bid_price):
                if (ask[1] != config.market_order_ask_price):
                    mcp = ask[1] + config.k*(-bid[1] - ask[1])
                else:
                    mcp = -bid[1] / (1.0 + config.default_margin)
            else:
                if (ask[1] != 0):
                    mcp = ask[1] * (1.0 + config.default_margin)

            if (transfer == bid[2]):                   # bid is fully cleared 
                asks_df['Quantity'][:1] = asks_df['Quantity'][:1] + transfer   # ask quantity is negative
                ask[2] = -transfer
                cleared_asks.append(ask)
                cleared_bids.append(bid)
                last_uncleared_ask = ask[1]
                bids_df = bids_df[1:]
            else:                                     # ask is fully cleared  
                bids_df['Quantity'][:1] = bids_df['Quantity'][:1] - transfer
                bid[2] = transfer
                cleared_bids.append(bid)
                cleared_asks.append(ask)
                asks_df = asks_df[1:]
                last_uncleared_ask = asks_df[:1].values[0][1]

        cleared_asks_df = pd.DataFrame(cleared_asks, columns=['ID', 'Price', 'Quantity'])
        cleared_bids_df = pd.DataFrame(cleared_bids, columns=['ID', 'Price', 'Quantity'])
        
        return mcp, total_mwh, cleared_asks_df, cleared_bids_df, last_uncleared_ask


    def reset(self):
        pass

    def step(self):
        pass

    def render(self):
        pass

    def close(self):
        pass