import sys
sys.path.insert(1, './ddpgbbs/gym_powertac')

from powertac_wm import PowerTAC_WM

import gym

import math
import datetime
import numpy as np
import pandas as pd

from gym import spaces

import tensorflow as tf

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU

model_storage_path = "../ddpg_v1.0" 

from config import Config
config = Config()

## DDPB based bidding strategy

class DDPGBBS(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4, "name": "DDPGBBS-v0"}

    # network parameters
    BUFFER_SIZE = 250000
    BATCH_SIZE = 32
    EPOCHS = 5
    GAMMA = 0.99
    TAU = 0.001     #Target Network HyperParameters
    LRA = 0.0001    #Learning rate for Actor
    LRC = 0.001     #Lerning rate for Critic

    '''
    Action : [
            limitprice1 belongs to R
            limitprice2 belongs to R
            ] (TWO output nuerons)
    '''
    action_dim = 2

    '''
    State : [
            Proximity (1)
            Balancing_Price (1)
            Required_Quantity (1)
            ] (total 26 input nuerons)
    '''
    state_dim = 3

    np.random.seed(1337)
    EXPLORE = 100000.0

    step = 0
    epsilon = 1

    ou = OU()       #Ornstein-Uhlenbeck Process

    def __init__(self, render_mode=None):
        self.seed_value = 0 # seed value will be overidden
        self.observation_space = spaces.Discrete(25)
        self.action_space = spaces.Box(-100.0, -10.0, shape=(1,), dtype=float)
        self.render_mode = render_mode
        self.type = "DDPGBBS"

        self.actor = ActorNetwork(self.state_dim, self.action_dim, self.BATCH_SIZE, self.TAU, self.LRA)
        self.critic = CriticNetwork(self.state_dim, self.action_dim, self.BATCH_SIZE, self.TAU, self.LRC)
        self.buff = ReplayBuffer(self.BUFFER_SIZE)    #Create replay buffer

        #Now load the weight
        print("Now we load the weight")
        try:
            self.actor.model.load_weights(model_storage_path + "/actormodel.h5")
            self.critic.model.load_weights(model_storage_path + "/criticmodel.h5")
            self.actor.target_model.load_weights(model_storage_path + "/actortargetmodel.h5")
            self.critic.target_model.load_weights(model_storage_path + "/critictargetmodel.h5")
            print("Weights loaded successfully")
        except Exception as e:
            print(e)
            print("Cannot find the weights")


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

            avg_buy_balancing_price = self.get_avg_buy_balancing_price()
            action = self.get_ddpg_action(proximity, avg_buy_balancing_price, rem_quantity)
            
            limit_price1 = -action[0]*avg_buy_balancing_price
            limit_price2 = -action[1]*avg_buy_balancing_price

            if proximity == 1:
                bids.append([self.id, -config.market_order_bid_price, rem_quantity/2])
                bids.append([self.id, -config.market_order_bid_price, rem_quantity/2])
            else:
                bids.append([self.id, limit_price1, rem_quantity/2])
                bids.append([self.id, limit_price2, rem_quantity/2])

        else:
            if proximity == 1:
                bids.append([self.id, config.market_order_bid_price, rem_quantity])

        if return_buyers_df != None:
            return_buyers_df[self.id] = bids
        else:
            return bids

    def get_ddpg_action(self, proximity, avg_buy_balancing_price, rem_quantity):
        
        norm_proximity = self.normalize(proximity, "Proximity")
        balancing_price = self.normalize(avg_buy_balancing_price, "BalancingPrice")
        quan = self.normalize(abs(rem_quantity), "Quantity")

        state = MDP_DDPG_State(norm_proximity, balancing_price, quan) 
        actions = list()

        try:
            s_t = np.array(state.form_network_input())
            a_t_original = self.actor.model.predict(s_t.reshape(1, s_t.shape[0]))[0].tolist()  # outputs a list of two limitprices
            actions.append(list(a_t_original))
        except:
            temp1 = np.random.rand()
            temp2 = np.random.rand()

            actions.append(max(temp1, temp2))
            actions.append(min(temp1, temp2))

        return actions
        

    def normalize(self, data, flag):
        
        if flag == "Proximity":
            return ((data - config.proximity_record_min) / (config.proximity_record_max - config.proximity_record_min))
        elif flag == "BalancingPrice":
            return ((data - config.balancing_price_record_min) / (config.balancing_price_record_max - config.balancing_price_record_min))
        elif flag == "Quantity":
            return ((data - config.quantity_record_min) / (config.quantity_record_max - config.quantity_record_min))                
    
    def get_avg_buy_balancing_price(self):
        pass

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


class MDP_State:   # not needed at the moment, if we decide to train new models, then complete this class

    class Exeperience:

        def __init__(self):
            self.state = None
            self.action = None     # new Pair<Double, Double> (null, null)
            self.reward = None
            self.nextState = None
            self.terminal = None

        def to_string(self):
            ret = '\n'
            ret += 'State ' + str(self.state) + '\n'
            ret += 'Action: ' + str(self.action) + '\n'
            ret += 'Reward: ' + str(self.reward )+ '\n'
            ret += 'Next State: ' + str(self.nextState )+ '\n'
            ret += 'Terminal: ' + str(self.terminal )+ '\n'

            return ret

    def __init__(self):
        self.exeperienceMap = dict()

        for i in range(1, 25):
            self.exeperienceMap.update({i: self.Exeperience()})


class MDP_DDPG_State:

    def __init__(self, proximity, balancingPrice, quantity):
        self.proximity = proximity
        self.balancingPrice = balancingPrice
        self.quantity = quantity

    def to_string(self):
        ret = '\n'
        ret += 'Proximity ' + str(self.proximity) + '\n'
        ret += 'True Valuation: ' + str(self.balancingPrice) + '\n'
        ret += 'Quantity: ' + str(self.quantity )+ '\n'

        return ret


gym.envs.register(
    id='DDPGBBS-v0',
    entry_point='broker.ddpgbbs:DDPGBBS',
)