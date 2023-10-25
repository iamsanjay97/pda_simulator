import functools
import random
from copy import copy
import math
import numpy as np
from gymnasium.spaces import Box, Dict

from pettingzoo.utils.env import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers
from collections import Counter
# from gymnasium.wrappers import FlattenObservation

import matplotlib.pyplot as plt

def env(render_mode=None):
    
    env = AuctionEnv()
    
    return env


def raw_env(render_mode=None):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = AuctionEnv(render_mode=render_mode)
    env = parallel_to_aec(env)
    return env


class AuctionEnv(ParallelEnv):
    metadata = { "render_modes": ["human"], "name": "AuctionEnv-v0",}

    def __init__(self,render_mode=None):

        # self.max_num_agents = 2
        self.num_of_agents = 3
        self.possible_agents = list(range(self.num_of_agents))
        self.timestep = 0
        self.horizon = 24 # Horizon (finite horizon game)
        self.seed_value = 10
        np.random.seed(self.seed_value)
        # Environment specific details (can it change in each iteration of the episode?)
        self.no_of_bids = 1
        self.max_bid_price = 500
        self.min_bid_price = 0.5
        self.min_bid_qty = 30
        self.max_bid_qty = 100
        self.max_req_quantity = 300 #Mwhr
        self.min_req_quantity = 30 #Mwhr
        self.balancing_price = 2 * self.max_bid_price # Psi price outisde the auction <---- Assumption
        self.requirements = np.round(np.random.uniform(self.min_req_quantity,self.max_req_quantity,size=(self.num_of_agents,)),2)
        # self.requirements = np.array([1500.0])
        self.supply_curve = NCpGenco()
        self.total_quantity = 1500
        self.asks = self.supply_curve.asks()

        ## asks for clearing mechanism
        self.last_cl_ask_index = 0
        

        self.no_of_asks = len(self.asks)
        self.max_ask_price = self.asks[self.no_of_asks-1,0]
        self.min_ask_price = self.asks[0,0]
        self.max_ask_qty = self.asks[0,1]
        self.min_ask_qty = self.supply_curve.min_ask_quant
        self.agents = copy(self.possible_agents)
        self.render_mode = render_mode

    def _clearing_mechanism(self,actions):
        # actions are bids convert to list of bids
        
        actions_li = {k : v.tolist() for k, v in actions.items()}
        
        # print(actions_li)
        # Seperate bids from agent and have a mapping of player to bids
        bid_vals = []
        bid_player_map = []

        for key in actions_li.keys():
            for val in actions_li[key]:
                bid_vals.append(val)
                bid_player_map.append(key)
        # self.bid_vals = bid_vals
        # print('here',bid_vals)
        
        bid_val_ind = sorted(range(len(bid_vals)), key=lambda k: (-bid_vals[k][0],-bid_vals[k][1]))
    
        bid_vals = [bid_vals[i] for i in bid_val_ind]    
        self.sorted_bids = np.array(bid_vals)

        if self.render_mode == "human":
            self.render()
        bid_player_map = [bid_player_map[i] for i in bid_val_ind]

        # Then sort the bids and the mapping
        uni_bids = {}
        for key, val in bid_vals:
            if key in uni_bids:
                uni_bids[key].extend([val])
            else:
                uni_bids[key] = [val] 
   
        # Unique price bids that is all bids with same are combined
        # Then pass the bids to combine the bids with the same price
        uni_bids_comb =[]
        for k,v in uni_bids.items():
            uni_bids_comb.append([k,sum(v)])

        # Then use the clearing mechanism for unique prices

        # copies need or not???
        # _asks = asks.copy()
        # _bids = uni_bids_comb.copy()
        # copy the ask and bid quantities
        cleared_asks_quant = [ask[1] for ask in self.asks]
        cleared_bids_quant = [bid[1] for bid in uni_bids_comb]

        # iterators and variables
        i = self.last_cl_ask_index
        j = 0
        N_a = len(self.asks)
        N_b = len(uni_bids_comb)
        # print(i,"i")
        # print("N_a = ",N_a,"N_b =" ,N_b )

        last_ask_index = self.last_cl_ask_index
        last_bid_index = N_b
        # print(last_ask_index,"last_ask_index   1")

        # print(uni_bids_comb,"bids")
        
        while j < N_b and i < N_a:
            if self.asks[i][0] > (uni_bids_comb[j][0]):
                break
            match_quant = min(self.asks[i][1],uni_bids_comb[j][1])
            self.asks[i][1] -= match_quant        
            uni_bids_comb[j][1] -= match_quant 
            last_ask_index, last_bid_index = i, j
            
        
            
            # Check if the quantity of particular ask is satisfied, 
            # if yes then go to next ask
            if self.asks[i][1] == 0:
                i += 1 


            # Check if the quantity of particular bid is satisfied, 
            # if yes then go to next bid
            if uni_bids_comb[j][1] == 0:
                j += 1
            
        # print(self.asks[:,1])
        # print(uni_bids_comb)
        # print(uni_bids_comb)

        cleared_asks_quant = cleared_asks_quant - self.asks[:,1]

        # print(cleared_asks_quant,"cleared ask quantity")
        # print(self.last_cl_ask_index,"cleared ask index")
        uni_bids_comb_column = [u[1] for u in uni_bids_comb]
        cleared_bids_quant = np.array(cleared_bids_quant) - np.array(uni_bids_comb_column)



        #Total cleared quantity
        mcq = sum(cleared_asks_quant) 

        # Clearing price
        if mcq == 0:
            self.mcp = 0 ## Need to verify this 
            
        # Average clearing price rule is used; However can be modified to get range of mcp
        else:
            self.mcp = (self.asks[last_ask_index][0]+uni_bids_comb[last_bid_index][0])/2 

        # print(last_ask_index,"last_ask_index   2")
        # Distribute the cleared quantitites among bids of same price (if any)

        cleared_bids = []
        prices = list(uni_bids.keys()) ## ERROR ALMOST SAME NAME NEED TO CHANGE

        # print(prices)
        for i in range(len(prices)):
        # Assumes that quantities are in descending order for the same price
            if cleared_bids_quant[i] == 0: # Price not cleared
                for _ in uni_bids[prices[i]]:
                    cleared_bids.append([prices[i],0])
                
            elif sum(uni_bids[prices[i]]) == cleared_bids_quant[i]: # Fully cleared
                for v in uni_bids[prices[i]]:
                    cleared_bids.append([prices[i],v])
            else: # Partially cleared; So need to distribute among equal price bids
                count = Counter(uni_bids[prices[i]])
                comb_quant = sorted(count.items(), key=lambda pair: pair[0], reverse=True)
        #         print(prices[i],comb_quant)

                clear_quant_price = cleared_bids_quant[i]
                for j in range(len(comb_quant)):
                    
                    if clear_quant_price == 0: 
                        # Not cleared
                        for _ in range(comb_quant[j][1]):
                            cleared_bids.append([prices[i],0])
                    elif comb_quant[j][0] * comb_quant[j][1] <=  clear_quant_price:
                        # Fully cleared
                        for _ in range(comb_quant[j][1]):
                            cleared_bids.append([prices[i],comb_quant[j][0]])
                        clear_quant_price -= comb_quant[j][0] * comb_quant[j][1]
                    else:
                        # Partially cleared and Equally distributed
                        each_cleared_quant = clear_quant_price/comb_quant[j][1] # round() is removed
                        
        #                 # To make sure the quantitity is not more than existing after rounding
        #                 if comb_quant[j][1] * each_cleared_quant > clear_quant_price:
        #                     each_cleared_quant = round(( - 1/100),2)
                        
                        for _ in range(comb_quant[j][1]):
                            cleared_bids.append([prices[i],each_cleared_quant])
                        #clear_quant_price -= comb_quant[j][0] * comb_quant[j][1] # might lead to numerical instability
                        # Hence set
                        clear_quant_price = 0
        # Calculate rewards or costs
        rewards = {}
        cleared_quant_agent = {}
        for key,  val in zip(bid_player_map,cleared_bids):
            if key in rewards:
                rewards[key] += (1 * self.mcp) * val[1] # <----- respresents cost hence negative
                cleared_quant_agent[key] += val[1]
            else:
                rewards[key] = (1* self.mcp) * val[1] # <------- Represents cost hence negative
                cleared_quant_agent[key] = val[1] 
        
        rewards = {k: v for k, v in sorted(list(rewards.items()))} # See better way to do this
        cleared_quant_agent = {k: v for k, v in sorted(list(cleared_quant_agent.items()))}

        # Change the index of the asks


        # print(last_ask_index,"last_ask_index   3")   
        
        # print(self.asks[last_ask_index][1],"cleared quantity")
        # print(last_ask_index,"last ask index")
        # print(N_a,"asks no")
        # print(self.asks[last_ask_index][1] == 0)
        # print(self.asks[last_ask_index][1] == 0.0)

        # ## Make the ask price to zero for the fully cleared ask

        # print(self.last_cl_ask_index,"prev last cl ind")
        # print(last_ask_index,"pres last cl ind")
        # print(self.asks,"asks")
        # for inx in range(self.last_cl_ask_index,last_ask_index+1):
        #     if self.asks[inx][1] == 0:
        #         self.asks[inx][0] = 0 


        # print(self.asks, "asks")
        if last_ask_index > -1 and last_ask_index < N_a:
            if self.asks[last_ask_index][1] == 0.0:
                self.last_cl_ask_index = last_ask_index + 1
                self.asks[last_ask_index][0] = 0
            else:
                self.last_cl_ask_index = last_ask_index


        # print(self.last_cl_ask_index,"cleared ask index later")
        
       
        # Update requirements
        # print("requirements before",self.requirements)
        for i in self.agents:
            self.requirements[i] = max(0.0,self.requirements[i]-cleared_quant_agent[i])
        
        # print("requirements",self.requirements)
        # print("cleared quantities",cleared_quant_agent)

        # print("cleared quant", cleared_quant_agent[i])
        # print("requirements",self.requirements)
        # print("timestep",self.timestep)
        return rewards

    



    def reset(self, seed=None, options=None):
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

    def step(self, actions):
        # Execute actions
        # print(actions,"actions")
        # if not actions:
        #     print(actions,"not actions")
        #     self.agents = []
        #     return {}, {}, {}, {}, {}
        

        # rewards = {} should be a dictionary of rewards for each agent

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
        # print(self.agents,"self.agents")
        # print('timestep',self.timestep)
        
                # self.requirements = np.delete(self.requirements,i)
                # print(terminations)


        # Check truncation conditions (overwrites termination conditions)
        
        # if self.timestep >=self.horizon:
        #     truncations = {i: True for i in range(self.num_agents)}
        #     self.agents = []
            # print(terminations)

        
        # truncations = {}
        # Get observations
        

        # Get dummy infos (not used in this example)
        

        return observations, rewards, terminations, truncations, infos

    def render(self):
        
        asks_q = self.asks[self.last_cl_ask_index:,1].cumsum()
        xmax = asks_q
        xmin = np.append(0,asks_q[:-1])


        bids_q = self.sorted_bids[:,1].cumsum()
        xmax_b = bids_q
        xmin_b = np.append(0,bids_q[:-1])

        # print(self.last_cl_ask_index,"index")

        # print(asks[self.last_cl_ask_index:,0],"ask---")
        plt.hlines(y = self.asks[self.last_cl_ask_index:,0], xmin = xmin, xmax = xmax, colors = "grey")
        plt.scatter(xmax, self.asks[self.last_cl_ask_index:,0], marker='D', s=10 ,c ="black")
        plt.hlines(y = self.sorted_bids[:,0], xmin = xmin_b, xmax = xmax_b, colors = "red")
        plt.scatter(xmax_b, self.sorted_bids[:,0], marker='o', s=10 ,c ="green")


        plt.grid(True, color = "grey", linewidth = "0.5", linestyle = "-.")

        
        plt.show()        

    def close(self):
        plt.close('all') # <<<< Need to see if this works
        

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