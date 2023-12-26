import copy
import sys  
import math
import numpy as np
import pandas as pd
import multiprocessing

import gym
import genco
import pdauction
import broker

from gym import spaces
from tqdm import tqdm
from collections import OrderedDict

sys.path.append('/home/sanjay/Research/MCTS/Codes/pda_simulator')
# sys.path.append('D:\PowerTAC\TCS\mcts\pda_simulator')
from config import Config

'''
State: Proximity
Action: Discretized actions for limit-price (total 7 actions)
Reward: Average limitprice at each step
'''

config = Config()

class MCTS_Vanilla(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4, "name": "MCTS_Vanilla-v0"}

    def __init__(self, render_mode=None):
        self.seed_value = 0 # seed value will be overidden
        self.observation_space = spaces.Discrete(25)
        self.action_space = spaces.Discrete(4)
        self.render_mode = render_mode
        self.type = "Discrete MCTS"

    # for discrete MCTS, buy_limit_price_max is set to be -10 as to use the values of both the limitprice fractions, 
    # may help placing smaller yet effective bids
    def set(self, total_demand, number_of_bids=1, buy_limit_price_min=-100.0, buy_limit_price_max=-10.0, sell_limit_price_min=0.5, sell_limit_price_max=90.0, id='MCTS'):
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

        self.action = Action()   
        self.root = TreeNode()  
        self.root.hour_ahead_auction = config.HOUR_AHEAD_AUCTIONS  

    def set_cleared_demand(self, cleared_demand):
        self.cleared_demand += cleared_demand
        
    def set_last_mcp(self, mcp):
        self.last_mcp = mcp

    def get_action_set(self):
        return self.action
    
    def set_supply(self, seller_quantities):
        self.supply = seller_quantities

    def set_demand(self, player_quantities):
        self.demand = player_quantities
    
    def set_quantities(self, player_total_demand):
        self.quantities = player_total_demand

    def update_buy_limit_price_max(self, price):
        # self.buy_limit_price_max = max(self.buy_limit_price_min, price)
        pass  # do not narrow the limitprice range, miso is selling so need to place lower limitprices

    def bids(self, timeslot, current_timeslot, return_buyers_df=None, random=False):

        rem_quantity = self.total_demand - self.cleared_demand
        
        if rem_quantity < self.min_bid_quant:
            if return_buyers_df != None:
                return_buyers_df[self.id] = None
            else:
                return None
        
        proximity = timeslot - current_timeslot                         # Proximity runs from 24 to 1 for a day-ahead PDA
        
        bids = list()
        # root = TreeNode()
        # self.root.hour_ahead_auction = proximity

        if rem_quantity > 0.0:
            if not random:
                for i in tqdm(range(config.NUMBER_OF_ROLLOUTS)): 
                    mcts = copy.deepcopy(self)
                    self.root.run_mcts(mcts, rem_quantity)

                best_move = self.root.best_action()
                self.root = best_move
                print("\nBest Move: ", best_move.to_string())
            else:
                best_move = self.root.default_policy(self)

            if(best_move != None):
                limit_price_range = [self.buy_limit_price_min, self.buy_limit_price_max]  
                limit_price = best_move.limit_price_fractions[0]*limit_price_range[0] + best_move.limit_price_fractions[1]*limit_price_range[1]
                bids.append([self.id, limit_price, rem_quantity])
        else:
            bids.append([self.id, config.market_order_bid_price, rem_quantity])

        if return_buyers_df != None:
            return_buyers_df[self.id] = bids
        else:
            return bids

    def get_limitprice(self, index):
        limit_price_range = [self.buy_limit_price_min, self.buy_limit_price_max]  
        best_fractions = self.get_action_set().get_limit_price_fractions(index)
        limit_price = best_fractions[0]*limit_price_range[0] + best_fractions[1]*limit_price_range[1]
        return limit_price
    
    def reset(self):
        pass

    def step(self):
        pass


class Action:
    
    def __init__(self, index=0): 
        self.index = index
        self.multiplier = np.array([[1.0, 0.0], [0.84, 0.16], [0.66, 0.33], [0.50, 0.50], [0.33, 0.66], [0.16, 0.84], [0.0, 1.0]])
        self.max_action = self.multiplier.shape[0]

    def get_action_index(self):
        return self.index
    
    def get_action_size(self):
        return self.max_action
    
    def get_limit_price(self, minPrice, maxPrice, index):
      factor = self.multiplier[index]
      return minPrice*factor[0] + maxPrice*factor[1]

    def get_limit_price_fractions(self, index):
      return self.multiplier[index]
    
    def get_wholesale_bid(self, minPrice, maxPrice, neededKwh):
      quotient = self.index 
      price = self.get_limit_price(minPrice, maxPrice, quotient)
      return price, neededKwh

    def set_action_index(self, index):
        self.index = index

    def to_string(self):
      return str(self.index)


class TreeNode:
    
    def __init__(self, id='MCTS_Vanilla', node=None):
        self.id = id
        self.epsilon = 1e-6
        self.alpha = 0.5
        self.K = 15       # multiplier to control exploration
        self.min_mwh = 0.001
        self.children = dict()
        
        self.n_visits = 0
        self.tot_value = 0

        self.hour_ahead_auction = 0
        self.applied_action = 0
        self.limit_price_fractions = 0
        
    
    def is_leaf(self, rem_quantity):
        return True if (self.hour_ahead_auction == 0 or len(self.children) == 0 or rem_quantity == 0) else False 
    
    
    def random_select(self, mcts):
        i = np.random.randint(0, mcts.get_action_set().get_action_size())
        node = TreeNode()
        node.hour_ahead_auction = self.hour_ahead_auction-1
        node.applied_action = i
        node.limit_price_fractions = mcts.get_action_set().get_limit_price_fractions(i)
        
        return i, node
    
    
    def uct_select(self, mcts):
        selected = None
        best_value = -1e9
        
        for child in self.children:
            n_visit_value = 0
            # total_point = 0

            if self.children.get(child).n_visits == 0:
                n_visit_value = 1 + self.epsilon
                # total_point = self.children.get(child).tot_value / n_visit_value
            else:
                n_visit_value = self.children.get(child).n_visits + self.epsilon
                # total_point = self.children.get(child).tot_value / n_visit_value
    
            visit_point = math.sqrt(2 * math.log(self.n_visits + 1) / n_visit_value)
            rand_point = np.random.random() * self.epsilon
            uct_value = self.children.get(child).tot_value + self.K*visit_point + rand_point

            if uct_value > best_value:
                selected = self.children.get(child)
                best_value = uct_value

        if selected == None:
            action, selected = self.random_select(mcts)                # action is the limitprice, next_state is the new TreeNode
            self.children.update({action: selected})

        return selected
    

    def select(self, mcts):

        if len(self.children) < mcts.get_action_set().get_action_size():
            action, next_state = self.random_select(mcts)             # action is the limitprice, next_state is the new TreeNode
            self.children.update({action: next_state})
        else:
            next_state = self.uct_select(mcts) 

        return next_state
    
    
    def step(self, limitprice, needed_mwh, list_of_sellers, list_of_buyers, pda):
        
        auction_proximity = self.hour_ahead_auction
        total_cost = 0.0
        total_quantity = 0.0

        # asks dataframe
        asks_df = pd.DataFrame(list_of_sellers['cp_genco'].asks(), columns=['ID', 'Price', 'Quantity'])

        # occasionally generate small random asks
        if (auction_proximity != config.HOUR_AHEAD_AUCTIONS) and (np.random.random() < 0.33):
            random_asks = pd.DataFrame([["miso", config.DEFAULT_MCP/10.0 + np.random.random()*0.7*config.DEFAULT_MCP, -np.random.normal(15, 1)]], columns=['ID', 'Price', 'Quantity'])
            asks_df = pd.concat([asks_df, random_asks], ignore_index=True)
            asks_df = asks_df.sort_values(by=['Price'])

        # bids dataframe
        if auction_proximity == config.HOUR_AHEAD_AUCTIONS:
            bids_df = pd.DataFrame([["miso", -1e9, np.random.normal(800, 100)]], columns=['ID', 'Price', 'Quantity'])
        else:
            bids_df = pd.DataFrame(columns=['ID', 'Price', 'Quantity'])

        for buyer in list_of_buyers.keys():
            if buyer == list(list_of_buyers.keys())[0]:
                buyer_df = pd.DataFrame([[list_of_buyers[buyer].id, limitprice, needed_mwh]], columns=['ID', 'Price', 'Quantity'])
            else:
                buyer_df = pd.DataFrame(list_of_buyers[buyer].bids(config.HOUR_AHEAD_AUCTIONS, config.HOUR_AHEAD_AUCTIONS - auction_proximity), columns=['ID', 'Price', 'Quantity'])
            bids_df = pd.concat([bids_df,buyer_df], ignore_index=True)

        bids_df = bids_df.sort_values(by=['Price'])

        # market clearing
        mcp, mcq, cleared_asks_df, cleared_bids_df, last_uncleared_ask = pda.clearing_mechanism(asks_df, bids_df)

        # update the cleared quantity of sellers
        for seller in list_of_sellers.keys():
            temp = cleared_asks_df.groupby('ID')
            if seller in temp.groups.keys():
                seller_cq = temp.sum()['Quantity'][seller]
                list_of_sellers[seller].set_cleared_quantity(-seller_cq)

        # update the cleared quantity of buyers
        buyer_cq = 0
        for buyer in list_of_buyers.keys():
            temp = cleared_bids_df.groupby('ID')
            if buyer in temp.groups.keys():
                buyer_cq = temp.sum()['Quantity'][buyer]
                list_of_buyers[buyer].set_cleared_demand(buyer_cq)
                list_of_buyers[buyer].set_last_mcp(mcp)
                
            if buyer == list(list_of_buyers.keys())[0]:
                total_cost = -(mcp*buyer_cq)
                total_quantity += buyer_cq
                needed_mwh -= buyer_cq     

        avg_clearing_price = -config.DEFAULT_MCP if total_quantity == 0 else total_cost/total_quantity     
                
        return avg_clearing_price, total_quantity, needed_mwh, list_of_sellers, list_of_buyers
    
    
    def simulation(self, needed_mwh, list_of_sellers, list_of_buyers, pda):
        
        auction_proximity = self.hour_ahead_auction

        rounds = config.HOUR_AHEAD_AUCTIONS
        cur_round = rounds - auction_proximity
        total_cost = 0.0
        avg_mcp = 0
        count = 0
        mcts_total_cleared_quantity = 0

        while(cur_round < rounds):
            
            proximity = rounds - cur_round       # runs from 23 to 1 (24 never happens in simulation)
            
            # asks dataframe
            asks_df = pd.DataFrame(list_of_sellers['cp_genco'].asks(), columns=['ID', 'Price', 'Quantity'])

            # occasionally generate small random asks
            if (auction_proximity != config.HOUR_AHEAD_AUCTIONS) and (np.random.random() < 0.33):
                random_asks = pd.DataFrame([["miso", config.DEFAULT_MCP/10.0 + np.random.random()*0.7*config.DEFAULT_MCP, -np.random.normal(15, 1)]], columns=['ID', 'Price', 'Quantity'])
                asks_df = pd.concat([asks_df, random_asks], ignore_index=True)
                asks_df = asks_df.sort_values(by=['Price'])

            # bids dataframe
            if proximity == config.HOUR_AHEAD_AUCTIONS:      
                bids_df = pd.DataFrame([["miso", -1e9, np.random.normal(800, 100)]], columns=['ID', 'Price', 'Quantity'])
            else:
                bids_df = pd.DataFrame(columns=['ID', 'Price', 'Quantity'])
                
            for buyer in list_of_buyers.keys():
                buyer_df = pd.DataFrame(list_of_buyers[buyer].bids(rounds, cur_round, random=True), columns=['ID', 'Price', 'Quantity'])
                bids_df = pd.concat([bids_df,buyer_df], ignore_index=True)

            # bids_df = pd.concat([bids_df,own_df], ignore_index=True)
            bids_df = bids_df.sort_values(by=['Price'])
                        
            # market clearing
            mcp, mcq, cleared_asks_df, cleared_bids_df, last_uncleared_ask = pda.clearing_mechanism(asks_df, bids_df)
            mcts_cleared_quantity = 0

            # update the cleared quantity of sellers
            asks_gb = cleared_asks_df.groupby('ID')
            for seller in list_of_sellers.keys():
                if seller in asks_gb.groups.keys():
                    seller_cq = asks_gb.sum()['Quantity'][seller]
                    list_of_sellers[seller].set_cleared_quantity(-seller_cq)
                
            # update the cleared quantity of buyers
            bids_gb = cleared_bids_df.groupby('ID')
            for buyer in list_of_buyers.keys():
                if buyer in bids_gb.groups.keys():
                    buyer_cq = bids_gb.sum()['Quantity'][buyer]
                    list_of_buyers[buyer].set_cleared_demand(buyer_cq)
                    list_of_buyers[buyer].set_last_mcp(mcp)

                    if buyer == list(list_of_buyers.keys())[0]:        
                        mcts_cleared_quantity = buyer_cq
                        mcts_total_cleared_quantity += mcts_cleared_quantity
            
            if mcq != 0:
                total_cost += -(mcp*mcts_cleared_quantity)
                avg_mcp = (avg_mcp*count + mcp) / (count+1)
                count += 1

            cur_round += 1

        rem_energy = needed_mwh - mcts_total_cleared_quantity

        b_price = 150.0 if (avg_mcp == 0.0) else 3*avg_mcp                      #  3 times the avg clearing price
        balancing_sim_sost = -abs(rem_energy) * b_price
        total_cost += balancing_sim_sost

        avg_clearing_price = -config.DEFAULT_MCP if needed_mwh == 0 else total_cost/needed_mwh

        return avg_clearing_price, needed_mwh
    
    
    def best_action(self):

        selected = None
        best_value = -1e9

        for child in self.children:
            total_point = self.children.get(child).tot_value

            if total_point > best_value:
                selected = self.children.get(child)
                best_value = total_point

        return selected
    
    
    def default_policy(self, mcts):
        i = np.random.randint(0, mcts.get_action_set().get_action_size())
        node = TreeNode()
        node.hour_ahead_auction = self.hour_ahead_auction-1
        node.applied_action = i
        node.limit_price_fractions = mcts.get_action_set().get_limit_price_fractions(i)
        return node
        
    
    def run_mcts(self, mcts, rem_quantity):
        
        visited = list()
        reward = list()
        cleared_quantity = list()
        cur_cost = 0
        visited.append(self)
        cur = self
        
        # prepare genco's asks and opponents' bids (in loop for multiple opponents)
        name_of_sellers = list()
        name_of_buyers = list()

        for item in mcts.supply.keys():
            name_of_sellers.append(item)

        for item in mcts.demand.keys():
            name_of_buyers.append(item)

        list_of_sellers = dict()
        list_of_buyers = dict()

        config = Config()
        pda = gym.make('pdauction/Auctioneer-v0')

        for seller in name_of_sellers:
            seller_obj = gym.make('genco/CPGenCo-v0')
            seller_obj.set_id(seller)
            seller_obj.set_cleared_quantity(mcts.supply[seller])
            list_of_sellers.update({seller: seller_obj})
            
        buyer1 = gym.make('MCTS_Vanilla-v0')
        buyer1.set(mcts.quantities[name_of_buyers[0]], 1, id=name_of_buyers[0])
        buyer1.set_cleared_demand(mcts.demand[name_of_buyers[0]])
        buyer2 = gym.make('ZI-v0')
        buyer2.set(mcts.quantities[name_of_buyers[1]], 1, id=name_of_buyers[1])
        buyer2.set_cleared_demand(mcts.demand[name_of_buyers[1]])

        list_of_buyers.update({name_of_buyers[0]: buyer1})
        list_of_buyers.update({name_of_buyers[1]: buyer2})
        
        while self.is_leaf(rem_quantity) == False:
            
            x_next = self.select(mcts)
            x_next_limitprice = mcts.get_limitprice(x_next.applied_action)
            r, q, rem_quantity, list_of_sellers, list_of_buyers = self.step(x_next_limitprice, rem_quantity, list_of_sellers, list_of_buyers, pda) # do a single auction
            reward.append(r)
            cleared_quantity.append(q)
            visited.append(x_next)
            self = x_next

        # expand  
        x_next = self.select(mcts)
        x_next_limitprice = mcts.get_limitprice(x_next.applied_action)
        r, q, rem_quantity, list_of_sellers, list_of_buyers = self.step(x_next_limitprice, rem_quantity, list_of_sellers, list_of_buyers, pda) # do a single auction
        reward.append(r)
        cleared_quantity.append(q)
        visited.append(x_next)
        self = x_next
        
        reward.append(0)    # acts as a padding to do calculation in a loop
        cleared_quantity.append(0)
        cur_cost, cur_quan = self.simulation(rem_quantity, list_of_sellers, list_of_buyers, pda)

        total = 0.0
        for iter,r,q in zip(reversed(visited),reversed(reward),reversed(cleared_quantity)):
            total = -config.DEFAULT_MCP if (cur_quan + q) == 0 else (cur_cost*cur_quan + r*q) / (cur_quan + q)
            cur_cost = total
            cur_quan += q
            iter.tot_value = (iter.tot_value*iter.n_visits + total) / (iter.n_visits+1)
            iter.n_visits += 1

        self = cur
            
        
    def to_string(self):

        ret = '\n'
        ret += 'Number of Visits: ' + str(self.n_visits) + '\n'
        ret += 'Total Value: ' + str(self.tot_value) + '\n'
        ret += 'Proximity: ' + str(self.hour_ahead_auction )+ '\n'
        ret += 'Applied Action: ' + str(self.limit_price_fractions) + '\n'

        return ret
    
gym.envs.register(
    id='MCTS_Vanilla-v0',
    entry_point='broker.mcts_vanilla_v0:MCTS_Vanilla',
)