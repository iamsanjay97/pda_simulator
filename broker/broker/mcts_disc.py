import sys  
import math
import numpy as np
import pandas as pd

import gym
import genco
import pdauction
import broker

from gym import spaces
from collections import OrderedDict

# sys.path.append('/home/sanjay/Research/MCTS/Codes/pda_simulator/broker/broker')
from config import Config

'''
State: Proximity
Action: Discretized actions for limit-price and quantity (total 49 actions)
Reward: Purchase cost at each step
'''

config = Config()

class MCTS_Disc(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4, "name": "MCTS_Disc-v0"}

    def __init__(self, render_mode=None):
        self.seed_value = 0 # seed value will be overidden

        self.observation_space = spaces.Box(0, 1, shape=(2,), dtype=float)
        self.action_space = spaces.Discrete(4)
        self.render_mode = render_mode

    def set(self, total_demand, number_of_bids=1, buy_limit_price_min=-100.0, buy_limit_price_max=-1.0, sell_limit_price_min=0.5, sell_limit_price_max=100.0, id='MCTS'):
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
        self.number_of_rollouts = config.NUMBER_OF_ROLLOUTS

        self.observer = Observer()
        self.action = Action()

    def set_cleared_demand(self, cleared_demand):
        self.cleared_demand += cleared_demand
        
    def set_last_mcp(self, mcp):
        self.last_mcp = mcp

    def get_action_set(self):
        return self.action

    def bids(self, timeslot, current_timeslot, random=False):

        rem_quantity = self.total_demand - self.cleared_demand
        
        if rem_quantity < self.min_bid_quant:
            return None
        
        remaining_tries = timeslot - current_timeslot

        self.observer.current_timeslot = current_timeslot
        self.observer.hour_ahead = remaining_tries
        self.observer.needed_enery_per_broker = rem_quantity
        self.observer.initial_needed_enery_mcts_broker = rem_quantity

        mcts = self
        root = TreeNode()
        root.hour_ahead_auction = self.observer.hour_ahead
        
        bids = list()
        if remaining_tries > 0:

            if rem_quantity > 0.0:

                if not random:
                    for i in range(self.number_of_rollouts): 
                        root.run_mcts(mcts.get_action_set(), mcts, self.observer)

                    best_move = root.final_select(self.observer)
                else:
                    best_move = root.select_random(mcts)

                # print("\nBest Move: ", best_move.to_string())

                if(best_move != None):
                    limit_price_range = [0, 100]    # TO DO: some way of getting this?
                    limit_price = best_move.limit_price_fractions[0]*limit_price_range[0] + best_move.limit_price_fractions[1]*limit_price_range[1]
                    quantity = best_move.quantity_ratio*rem_quantity
                    bids.append([self.id, limit_price, quantity])
            else:
                bids.append([self.id, config.market_order_bid_price, rem_quantity])
        else:
            bids.append([self.id, -config.market_order_bid_price, rem_quantity])

        return bids
    
    def reset(self):
        pass

    def step(self):
        pass
    

class Observer:

    def __init__(self):
        self.current_timeslot = 0
        self.hour_ahead = 0
        self.initial_needed_enery_mcts_broker = 0
        self.needed_enery_per_broker = self.initial_needed_enery_mcts_broker
        self.balacing_price = 90.0


class Action:
    
    def __init__(self, index=0): 
        self.index = index
        self.quantityRatio = np.array([0.0, 0.16, 0.33, 0.5, 0.66, 0.84, 1.0])
        self.multiplier = np.array([[1.0, 0.0], [0.84, 0.16], [0.66, 0.33], [0.50, 0.50], [0.33, 0.66], [0.16, 0.84], [0.0, 1.0]])
        self.max_action = self.quantityRatio.shape[0] * self.multiplier.shape[0]

    def get_action_index(self):
        return self.index
    
    def get_action_size(self):
        return self.max_action
    
    def get_qauntity_action_size(self):
        return self.quantityRatio.shape[0]
    
    def get_limit_price(self, minPrice, maxPrice, index):
      factor = self.multiplier[index]
      return minPrice*factor[0] + maxPrice*factor[1]
    
    def get_quantity_to_bid(self, index, neededKwh):
      return self.quantityRatio[index] * neededKwh

    def get_limit_price_fractions(self, index):
      quotient = int(index / self.quantityRatio.shape[0])
      return self.multiplier[quotient]

    def get_quantity_ratio(self, index):
      reminder = index % self.quantityRatio.shape[0]
      return self.quantityRatio[reminder]
    
    def get_wholesale_bid(self, minPrice, maxPrice, neededKwh):
      quotient = self.index / self.quantityRatio.shape[0]
      reminder = self.index % self.quantityRatio.shape[0]

      price = self.get_limit_price(minPrice, maxPrice, quotient)
      quantity = self.get_quantity_to_bid(reminder, neededKwh)
      return price, quantity

    def set_action_index(self, index):
        self.index = index

    def to_string(self):
      return str(self.index)
    

class TreeNode:

    def __init__(self, id='MCTS_Desc', node=None):
        self.epsilon = 1e-6
        self.min_mwh = 0.001
        self.fixed_vol = 50.0
        self.id = id

        self.children = dict()

        if node == None:
            self.n_visits = 0
            self.tot_value = 0
            self.current_node_cost = 0

            self.hour_ahead_auction = 0
            self.applied_action = 0
            self.limit_price_fractions = 0
            self.quantity_ratio = 0
        else:
            self.n_visits = node.n_visits
            self.tot_value = node.tot_value
            self.current_node_cost = node.current_node_cost

            self.hour_ahead_auction = node.hour_ahead_auction
            self.applied_action = node.applied_action
            self.limit_price_fractions = node.limit_price_fractions 
            self.quantity_ratio = node.quantity_ratio

    
    def unvisited_children(self, tn):
        count = 0

        for i in tn.children:
            if tn.children.get(i).n_visits == 0:
                count += 1

        return count
    

    def run_mcts(self, action, mcts, ob):
        needed_energy = ob.needed_enery_per_broker
        ini_needed_energy = ob.initial_needed_enery_mcts_broker
        visited = list()

        cur = self
        visited.append(cur)

        if(cur.is_leaf()):
            print('cur.isLeaf() is 0! This should not happen!')

        avg_mcp = 0.0
        count_clearing = 0
        
        while (cur.is_leaf() == False) and (needed_energy != 0):
            if len(cur.children) == 0:
                cur.expand(mcts)

            unvisited_children = self.unvisited_children(cur)

            if (unvisited_children == (action.get_action_size())):              # If this is first visit select random child
                cur = cur.select_random(mcts)
            else:
                balancing_price = 150.0 if(avg_mcp == 0.0) else 3*avg_mcp      
                cur = cur.select(mcts, ob, balancing_price)                     # UCT based selection
            visited.append(cur)

        print('\nStarting the Simulation ...')
        sim_cost = self.simulation(cur, ob, needed_energy, ini_needed_energy)
        print('\nSimulation Done ! \nSimulation Cost: ', sim_cost)

        for iter in reversed(visited):
            sim_cost += iter.current_node_cost   # TO DO: see if i'm double counting for the last node?
            iter.update_stats(sim_cost)
            

    def select_random(self, mcts):
        i = np.random.randint(0, mcts.get_action_set().get_action_size())
        return self.children.get(i)
    

    def expand(self, mcts):
        n_actions = mcts.get_action_set().get_action_size()
        self.children = dict()
        new_hour_ahead_auction = self.hour_ahead_auction - 1

        for i in range(n_actions):
            if (i == 0) or (i % mcts.get_action_set().get_qauntity_action_size() != 0):
                tn = TreeNode()
                tn.hour_ahead_auction = new_hour_ahead_auction
                tn.applied_action = i
                tn.limit_price_fractions = mcts.get_action_set().get_limit_price_fractions(i)
                tn.quantity_ratio = mcts.get_action_set().get_quantity_ratio(i)
                self.children.update({i: tn})
            else:
                self.children.update({i: self.children.get(0)})


    def final_select(self, ob):
        selected = None
        best_value = -1e9

        for child in self.children:
            total_point = self.children.get(child).tot_value

            if total_point > best_value:
                selected = self.children.get(child)
                best_value = total_point

        return selected


    def select(self, mcts, ob, balancing_price):
        selected = None
        best_value = -1e9
    
        for child in self.children:
            n_visit_value = 0
            total_point = 0

            if self.children.get(child).n_visits == 0:
                n_visit_value = 1 + self.epsilon
                total_point = self.children.get(child).tot_value
            else:
                n_visit_value = self.children.get(child).n_visits + self.epsilon
                total_point = self.children.get(child).tot_value
    
            dividend = -balancing_price * abs(ob.initial_needed_enery_mcts_broker)
            total_point = 1 - total_point / dividend

            visit_point = math.sqrt(2 * math.log(self.n_visits + 1) / n_visit_value)
            rand_point = np.random.random() * self.epsilon
            uct_value = total_point + visit_point + rand_point

            if uct_value > best_value:
                selected = self.children.get(child)
                best_value = uct_value

        return selected


    def simulation(self, tn, ob, needed_mwh, ini_needed_energy):
        bids = list()
        auction_proximity = self.hour_ahead_auction

        # prepare own bids
        limit_price_range = [0, 100]    # TO DO: some way of getting this?
        limit_price = tn.limit_price_fractions[0]*limit_price_range[0] + tn.limit_price_fractions[1]*limit_price_range[1]
        quantity = tn.quantity_ratio*needed_mwh

        if(quantity > self.min_mwh):
            bids.append([self.id, limit_price, quantity])
        
        # prepare genco's asks and opponents' bids (in loop for multiple opponents)
        name_of_sellers = ['cp_genco']
        name_of_buyers = ['ZI', 'MCTS_D']

        list_of_sellers = dict()
        list_of_buyers = dict()

        config = Config()
        pda = gym.make('pdauction/Auctioneer-v0')

        for seller in name_of_sellers:
            seller_obj = gym.make('genco/CPGenCo-v0')
            seller_obj.set_id('cp_genco')
            list_of_sellers.update({seller: seller_obj})
            
        buyer1 = gym.make('ZI-v0')
        buyer1.set(config.market_demand*0.7, 1, id=name_of_buyers[0])
        buyer2 = gym.make('MCTS_Disc-v0')
        buyer2.set(config.market_demand*0.3, 1, id=name_of_buyers[1])

        list_of_buyers.update({name_of_buyers[0]: buyer1})
        list_of_buyers.update({name_of_buyers[1]: buyer2})

        rounds = auction_proximity
        cur_round = 0
        total_cost = 0.0
        avg_mcp = 0
        count = 0
        mcts_total_cleared_quantity = 0

        while(cur_round < rounds):
            
            proximity = rounds - cur_round    
            
            # asks dataframe
            asks_df = pd.DataFrame(list_of_sellers['cp_genco'].asks(), columns=['ID', 'Price', 'Quantity'])

            # bids dataframe
            if proximity == config.HOUR_AHEAD_AUCTIONS:
                bids_df = pd.DataFrame([["miso", -1e9, np.random.normal(800, 100)]], columns=['ID', 'Price', 'Quantity'])
            else:
                bids_df = pd.DataFrame(columns=['ID', 'Price', 'Quantity'])
                
            for buyer in list_of_buyers.keys():
                buyer_df = pd.DataFrame(list_of_buyers[buyer].bids(rounds, cur_round+1, random=True), columns=['ID', 'Price', 'Quantity'])
                bids_df = pd.concat([bids_df,buyer_df], ignore_index=True)

            bids_df = bids_df.sort_values(by=['Price'])
                        
            # market clearing
            mcp, mcq, cleared_asks_df, cleared_bids_df = pda.clearing_mechanism(asks_df, bids_df)
            
            # update the cleared quantity of sellers
            for seller in list_of_sellers.keys():
                temp = cleared_asks_df.groupby('ID')
                if seller in temp.groups.keys():
                    seller_cq = temp.sum()['Quantity'][seller]
                    list_of_sellers[seller].set_cleared_quantity(-seller_cq)
                
            # update the cleared quantity of buyers
            for buyer in list_of_buyers.keys():
                temp = cleared_bids_df.groupby('ID')
                if buyer in temp.groups.keys():
                    buyer_cq = temp.sum()['Quantity'][buyer]
                    list_of_buyers[buyer].set_cleared_demand(buyer_cq)
                    list_of_buyers[buyer].set_last_mcp(mcp)

                    if buyer == "MCTS_D":                   # TO DO: add a type information into buyer
                        mcts_cleared_quantity = buyer_cq
                        mcts_total_cleared_quantity += mcts_cleared_quantity
            
            print('\n----------During Rollout: At Proxomity ', proximity, '------\n')
            if mcq != 0:
                print('CP', mcp)
                print('CQ', mcts_cleared_quantity)

                total_cost += mcp*mcts_cleared_quantity
                avg_mcp = (avg_mcp*count + mcp) / (count+1)
                count += 1

            cur_round += 1

        rem_energy = needed_mwh - mcts_total_cleared_quantity

        b_price = 150.0 if (avg_mcp == 0.0) else 3*avg_mcp                      #  3 times the avg clearing price
        balancing_sim_sost = abs(rem_energy) * b_price
        print("Remaining: ", rem_energy, " :: Balacing: ", balancing_sim_sost)
        total_cost += balancing_sim_sost
        tn.current_node_cost = -total_cost

        return total_cost


    def is_leaf(self):
        return True if self.hour_ahead_auction == 0 else False       # No more chances
    

    def update_stats(self, sim_cost):
        self.tot_value = (self.tot_value * self.n_visits + sim_cost) / (self.n_visits + 1)
        self.n_visits += 1

    
    def to_string(self):
        ret = '\n' # 'Tree Node Data' + '\n'

        ret += 'Number of Visits: ' + str(self.n_visits) + '\n'
        ret += 'Total Value: ' + str(self.tot_value) + '\n'
        ret += 'Proximity: ' + str(self.hour_ahead_auction )+ '\n'
        ret += 'Applied Action Index: ' + str(self.applied_action) + '\n'

        return ret
    

gym.envs.register(
    id='MCTS_Disc-v0',
    entry_point='broker.mcts_disc:MCTS_Disc',
)