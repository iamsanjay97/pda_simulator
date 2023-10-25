import gym
from gym import spaces
import pygame
import numpy as np
import functools


class AuctioneerEnv(gym.Env):
    metadata = {"render_modes": ["human"], "name": "Auctioneer-v0"}

    def __init__(self, number_of_agents, render_mode=None):

        self.default_margin = 0.05                # used when one side has no limit price
        self.default_clearing_price = 40.00       # used when no limit prices
        self.seller_surplus_ratio = 0.5
        self.seller_max_margin = 0.05
        self.mkt_posn_limit_initial = 90.0
        self.mkt_posn_limit_final = 143.0

        self.number_of_agents = number_of_agents
        self.horizon = 24                         # finite horizon of 24 hours

        self.number_of_bids = 1
        self.max_bid_price = 500
        self.min_bid_price = 0.5
        self.min_bid_qty = 0.01                   # in MwH
        self.max_bid_qty = 100                    # in MwH

        self.render_mode = render_mode


    def clearing_mechanism(self, actions):
        '''
            A funtion to perform the market clearing
        '''


    def reset(self, seed=None, options=None):
        '''
            A fuction to reset the environment
        '''
        self.agents = copy(self.possible_agents)
        # self.max_num_agents = len(self.agents)
        self.timestep = 0
        
        self.requirements = np.round(np.random.uniform(self.min_req_quantity,self.max_req_quantity,size=(self.num_of_agents,)),2)
        
        # self.requirements = np.array([1500.0])
        self.supply_curve = NCpGenco()
        self.asks = self.supply_curve.asks()
        self.last_cl_ask_index = 0
        self.no_of_asks = len(self.asks)
        self.max_ask_price = self.asks[self.no_of_asks-1,0]
        self.min_ask_price = self.asks[0,0]
        self.max_ask_qty = self.asks[0,1]
        self.min_ask_qty = self.supply_curve.min_ask_quant

        # No information initially for partial information game <---- To Do later

        # For complete information game can share requirements of all players 

        observation = {"ask_price" : self.asks[:,0] , "ask_qty": self.asks[:,1],"reqs" : self.requirements}
        observations = { i: observation for i in self.agents }
        return observations, {}


    def step(self, action):
        # actions are nothing but bids with keys as player id 
        if self.timestep < self.horizon:
            rewards = self._clearing_mechanism(actions)
        else:
            rewards = {i : self.balancing_price * self.requirements[i]  for i in self.agents}
        
        # print(self.timestep)
        # Check termination conditions
        observation = {"ask_price" : self.asks[:,0] , "ask_qty": self.asks[:,1],"reqs" : self.requirements}
        observations = { i: observation for i in self.agents }
        truncations = {i : False for i in self.agents}
        infos = {i:{} for i in self.agents}
        terminations = {i: False for i in self.agents}
        # print(terminations)
        

        # if self.timestep >= self.horizon-1:
        #     terminations = {i: True for i in range(self.num_agents)}
        #     self.agents = []

        for i in self.possible_agents:
            # print("env:agent",i)
            # print("env:requirements",self.requirements)
            if i in self.agents:
                if self.requirements[i] == 0.0 or self.timestep == self.horizon:
                    # print("inside")
                    terminations[i] = True
                    self.agents.remove(i)
                    # print("env:agents",self.agents)
                    # print("env:terminations",terminations)
                # self.requirements = np.delete(self.requirements,i)
        # print(terminations)
        # print("-----------------------")
        # print("self.agents",self.agents)

        self.timestep += 1

        return observations, rewards, terminations, truncations, infos


    def render(self):
        pass


    def close(self):
        pass


    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        ask_price = Box(self.min_ask_price,self.max_ask_price,shape=(self.no_of_asks,),dtype=np.float32)
        ask_qty = Box(self.min_ask_qty,self.max_ask_qty,shape=(self.no_of_asks,),dtype=np.float32)
        # ask_space = Tuple((ask_price,ask_qty))
        qty_reqs = Box(self.min_req_quantity,self.max_req_quantity,shape=(self.num_agents,),dtype=np.float32) 
    
        return Dict({"ask_price" : ask_price , "ask_qty": ask_qty,"reqs" : qty_reqs})
         

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        # bid_price = Box(self.min_bid_price,self.max_bid_price,shape=(),dtype=np.float32)
        # bid_qty = Box(self.min_bid_qty,self.max_req_quantity,shape=(),dtype=np.float32)
        action_space = Box(low=np.tile(np.array([self.min_bid_price, self.min_bid_qty]),(self.no_of_bids,1)), high=np.tile(np.array([self.max_bid_price, self.max_bid_qty]),(self.no_of_bids,1)), dtype=np.float32)
        return action_space