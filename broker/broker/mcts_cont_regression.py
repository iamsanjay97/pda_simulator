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
from sortedcontainers import SortedDict

# sys.path.append('/home/sanjay/Research/MCTS/Codes/pda_simulator')
sys.path.append('D:\PowerTAC\TCS\mcts\pda_simulator')
from config import Config

'''
State: Proximity
Action: Continuous actions for limit-price using Simple Progressive Widening
Reward: Average limitprice at each step
'''

config = Config()

class MCTS_Cont_Regression(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4, "name": "MCTS_Cont_Regression-v0"}

    def __init__(self, render_mode=None):
        self.seed_value = 0 # seed value will be overidden
        self.observation_space = spaces.Discrete(25)
        self.action_space = spaces.Box(-100.0, -10.0, shape=(1,), dtype=float)
        self.render_mode = render_mode
        self.type = "Continuous MCTS Regression"

    def set(self, total_demand, number_of_bids=1, buy_limit_price_min=-100.0, buy_limit_price_max=-1.0, sell_limit_price_min=0.5, sell_limit_price_max=90.0, id='MCTS'):
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
        self.action_space = spaces.Box(buy_limit_price_min, buy_limit_price_max, shape=(1,), dtype=float)
        self.root = TreeNode()  
        self.root.hour_ahead_auction = config.HOUR_AHEAD_AUCTIONS 

        self.auction_data = dict()

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

    # this gets updated based on the actual auction dynamics, contains actual mcp and mcp
    def update_auction_data(self, proximity, cp, cq):
        if proximity not in self.auction_data.keys():
            temp_dict = SortedDict({cp: cq})
        else:
            temp_dict = self.auction_data[proximity]
            temp_dict.setdefault(cp, cq)
        self.auction_data.update({proximity: temp_dict})

    def bids(self, timeslot, current_timeslot, return_buyers_df=None, random=False):

        rem_quantity = self.total_demand - self.cleared_demand
        
        if rem_quantity < self.min_bid_quant:
            if return_buyers_df != None:
                return_buyers_df[self.id] = None
            else:
                return None
        
        proximity = timeslot - current_timeslot                         # Proximity runs from 24 to 1 for a day-ahead PDA
        
        bids = list()
        # root = TreeNode()                 # keeping a same tree throughout 24 proxmities like alphaGo
        # self.root.hour_ahead_auction = proximity
        # print('MCTS SPW called')

        if rem_quantity > 0.0:
            if not random:
                for i in tqdm(range(config.NUMBER_OF_ROLLOUTS)): 
                    # mcts = copy.deepcopy(self)        # TO DO: No need to create a copy, mcts is object is not modified in run_mcts CHECK THIS
                    self.root.run_mcts(self, rem_quantity)

                # parallelizing simulation using multiprocessing 
                # with multiprocessing.Pool() as pool: 
                #     mcts = copy.deepcopy(self)
                #     pool.map(root.run_mcts, mcts, rem_quantity) 

                self.root = self.root.best_action()
                best_limitprice = self.root.applied_action_lp           # limitprice is negative
                print("\nBest Move: ", best_limitprice)
            else:
                best_limitprice = self.root.default_policy(self)      # limitprice is negative

            if(best_limitprice != None):
                bids.append([self.id, best_limitprice, rem_quantity])

        else:
            bids.append([self.id, config.market_order_bid_price, rem_quantity])

        if return_buyers_df != None:
            return_buyers_df[self.id] = bids
        else:
            return bids

    def reset(self):
        pass

    def step(self):
        pass


class TreeNode:
    
    def __init__(self, id='MCTS_Cont_Regression'):
        self.id = id
        self.epsilon = 1e-6
        self.alpha = 0.5
        self.K = 15       # multiplier to control exploration
        self.min_mwh = 0.001
        self.children = dict()
        
        self.n_visits = 0
        self.tot_cost = 0   # weighted average clearing price for the simulation below this node (negative value as it is a cost)

        self.hour_ahead_auction = 0
        self.applied_action_lp = 0
        self.p_cleared = 0.0  # clearing probability for the applied_action_lp
        
    
    # this gets updates within simulation using actual auction data, but for limitprices 
    # placed during the simulation phase
    def get_p_cleared(self, proximity, lp, auction_data):

        if proximity not in auction_data.keys():
            return -1
        
        data = auction_data[proximity]
        
        total_cleared = sum(data.values())
        this_cleared = 0.0

        for item in data:
            if abs(lp) > item:
                this_cleared += data[item]
            else:
                break

        if ((this_cleared == 0) or (this_cleared == total_cleared)) and len(auction_data[proximity]) < 5:
            return -1

        if total_cleared != 0:
            return this_cleared/total_cleared
        else:
            return 0

    
    def is_leaf(self, rem_quantity):
        return True if (self.hour_ahead_auction == 0 or len(self.children) == 0 or rem_quantity == 0) else False 
    
    
    # modified selection function
    # we check proximity of the auction, sample a limiprices and calculate their p_cleared
    # and pick the limitprice which is in the desirable range of p_cleared values
    # proximity 24 to 17 -> p_cleared [0.05, 0.50]
    # proximity 18 to 7 -> p_cleared [0.25, 0.75]
    # proximity 6 to 1 -> p_cleared [0.75, 1.0]
    def random_select(self, mcts, proximity):

        lp = None
        i = 0
        while True:
            lp = np.random.uniform(mcts.buy_limit_price_min, mcts.buy_limit_price_max)
            prob = self.get_p_cleared(proximity, lp, mcts.auction_data)

            if prob == -1:
                break

            if ((proximity > 18) and (proximity <= 24)) and ((prob > 0.05) and (prob <= 0.6)):
                break

            if ((proximity > 6) and (proximity <= 18)) and ((prob > 0.25) and (prob <= 0.85)):
                break

            if ((proximity > 0) and (proximity <= 6)) and ((prob > 0.75) and (prob <= 1.0)):
                break

            i += 1

            if i > 100:
                break

        node = TreeNode()
        node.hour_ahead_auction = self.hour_ahead_auction-1
        node.applied_action_lp = lp
        
        return lp, node
    
    
    def uct_select(self):
        selected = None
        best_value = -1e9
        
        for child in self.children:
            n_visit_value = 0
            # total_point = 0

            if self.children.get(child).n_visits == 0:
                n_visit_value = 1 + self.epsilon
                # total_point = self.children.get(child).tot_cost / n_visit_value
            else:
                n_visit_value = self.children.get(child).n_visits + self.epsilon
                # total_point = self.children.get(child).tot_cost / n_visit_value
    
            visit_point = math.sqrt(2 * math.log(self.n_visits + 1) / n_visit_value)
            rand_point = np.random.random() * self.epsilon
            uct_value = self.children.get(child).tot_cost + self.K*visit_point + rand_point

            if uct_value > best_value:
                selected = self.children.get(child)
                best_value = uct_value

        return selected
    
    
    def select(self, mcts, proximity):
                
        if math.pow(self.n_visits, self.alpha) >= len(self.children):
            action, next_state = self.random_select(mcts, proximity)                # action is the limitprice, next_state is the new TreeNode
            self.children.update({action: next_state})
        else:
            next_state = self.uct_select() 

        return next_state
    
    
    def step(self, mcts, limitprice, needed_mwh, list_of_sellers, list_of_buyers, pda):
        
        auction_proximity = self.hour_ahead_auction
        total_cost = 0.0
        total_quantity = 0.0

        # asks dataframe
        asks_df = pd.DataFrame(list_of_sellers['cp_genco'].asks(), columns=['ID', 'Price', 'Quantity'])

        # occasionally generate small random asks
        if (auction_proximity != config.HOUR_AHEAD_AUCTIONS) and (np.random.random() < 0.33):
            random_asks = pd.DataFrame([["miso", config.DEFAULT_MCP/20.0 + np.random.random()*0.4*config.DEFAULT_MCP, -np.random.normal(15, 1)]], columns=['ID', 'Price', 'Quantity'])
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
        if mcts.type == 'Continuous MCTS Regression':
            mcts.update_auction_data(auction_proximity, mcp, mcq)

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
    
    
    def simulation(self, mcts, needed_mwh, list_of_sellers, list_of_buyers, pda):
        
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
                random_asks = pd.DataFrame([["miso", config.DEFAULT_MCP/20.0 + np.random.random()*0.4*config.DEFAULT_MCP, -np.random.normal(15, 1)]], columns=['ID', 'Price', 'Quantity'])
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
            if mcts.type == 'Continuous MCTS Regression':
                mcts.update_auction_data(proximity, mcp, mcq)

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
        balancing_sim_cost = -abs(rem_energy) * b_price
        total_cost += balancing_sim_cost

        avg_clearing_price = -config.DEFAULT_MCP if needed_mwh == 0 else total_cost/needed_mwh

        return avg_clearing_price, needed_mwh
    
    
    def best_action(self):
        
        selected = None
        best_lp = -1e9
        
        for child in self.children:
            if self.children.get(child).tot_cost > best_lp:
                selected = self.children.get(child)
                best_lp = self.children.get(child).tot_cost

        return selected
    
    
    def default_policy(self, mcts):
        return np.random.uniform(mcts.buy_limit_price_min, mcts.buy_limit_price_max)
        
    
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
            
        buyer1 = gym.make('MCTS_Cont_Regression-v0')
        buyer1.set(mcts.quantities[name_of_buyers[0]], 1, id=name_of_buyers[0])
        buyer1.set_cleared_demand(mcts.demand[name_of_buyers[0]])
        buyer2 = gym.make('ZI-v0')
        buyer2.set(mcts.quantities[name_of_buyers[1]], 1, id=name_of_buyers[1])
        buyer2.set_cleared_demand(mcts.demand[name_of_buyers[1]])

        list_of_buyers.update({name_of_buyers[0]: buyer1})
        list_of_buyers.update({name_of_buyers[1]: buyer2})

        while self.is_leaf(rem_quantity) == False:

            x_next = self.select(mcts, self.hour_ahead_auction)
            prob = self.get_p_cleared(self.hour_ahead_auction, x_next.applied_action_lp, mcts.auction_data)
            prob = 1 if (prob == -1 ) else prob
            x_next.p_cleared = prob
            r, q, rem_quantity, list_of_sellers, list_of_buyers = self.step(mcts, x_next.applied_action_lp, rem_quantity, list_of_sellers, list_of_buyers, pda) # do a single auction
            reward.append(r)
            cleared_quantity.append(q)
            visited.append(x_next)
            self = x_next

        # expand  
        x_next = self.select(mcts, self.hour_ahead_auction)
        prob = self.get_p_cleared(self.hour_ahead_auction, x_next.applied_action_lp, mcts.auction_data)
        prob = 1 if (prob == -1 )else prob
        x_next.p_cleared = prob
        r, q, rem_quantity, list_of_sellers, list_of_buyers = self.step(mcts, x_next.applied_action_lp, rem_quantity, list_of_sellers, list_of_buyers, pda) # do a single auction
        reward.append(r)
        cleared_quantity.append(q)
        visited.append(x_next)
        self = x_next
        
        reward.append(0)    # acts as a padding to do calculation in a loop
        cleared_quantity.append(0)
        cur_cost, cur_quan = self.simulation(mcts, rem_quantity, list_of_sellers, list_of_buyers, pda)

        total = 0.0
        for iter,r,q in zip(reversed(visited),reversed(reward),reversed(cleared_quantity)):
            total = -config.DEFAULT_MCP if (cur_quan + q) == 0 else (cur_cost*cur_quan + r*q) / (cur_quan + q)
            cur_cost = total
            cur_quan += q
            iter.tot_cost = (iter.tot_cost*iter.n_visits + total) / (iter.n_visits+1)
            iter.n_visits += 1

        self = cur
            
        
    def to_string(self):

        ret = '\n'
        ret += 'Number of Visits: ' + str(self.n_visits) + '\n'
        ret += 'Total Value: ' + str(self.tot_cost) + '\n'
        ret += 'Proximity: ' + str(self.hour_ahead_auction )+ '\n'
        ret += 'Applied Action Index: ' + str(self.applied_action_lp) + '\n'

        return ret
    
gym.envs.register(
    id='MCTS_Cont_Regression-v0',
    entry_point='broker.mcts_cont_regression:MCTS_Cont_Regression',
)