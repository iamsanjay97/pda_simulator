import gym
import math
import numpy as np
import pandas as pd
import multiprocessing

import genco
import pdauction
from broker import zi, zip, mcts_vanilla_v0, mcts_vanilla_v1, mcts_cont_spw, mcts_cont_regression

from config import Config
from collections import OrderedDict

'''
Auction Handler contains:
    - Set of Sellers
    - Set of Buyers (MISO and other configured buyers)
    - Some random asks in between simulating MISO's selling pattern
    - Auction Clearing Mechanism
    - Data structures to maintain Sellers' and buyers' data
'''

# --------------------------------- Update this based on Configuration --------------------------------------- #

name_of_sellers = ['cp_genco']
# name_of_buyers = ['MCTS_Cont', 'MCTS_Vanilla', 'ZI']
name_of_buyers = ['MCTS-Cont-Reg', 'ZI']

list_of_sellers = dict()
list_of_buyers = dict()

config = Config()
pda = gym.make('pdauction/Auctioneer-v0')

for seller in name_of_sellers:
    seller_obj = gym.make('genco/CPGenCo-v0')
    seller_obj.set_id('cp_genco')
    list_of_sellers.update({seller: seller_obj})

buyers = [None]*len(name_of_buyers)
    
buyers[0] = gym.make('MCTS_Cont_Regression-v0')
buyers[0].set(config.market_demand*0.51, 1, id=name_of_buyers[0])
buyers[1] = gym.make('ZI-v0')
buyers[1].set(config.market_demand*0.49, 1, id=name_of_buyers[1])

# buyers[0] = gym.make('ZI-v0')
# buyers[0].set(config.market_demand*0.34, 1, id=name_of_buyers[0])
# buyers[1] = gym.make('ZI-v0')
# buyers[1].set(config.market_demand*0.33, 1, id=name_of_buyers[1])
# buyers[2] = gym.make('ZI-v0')
# buyers[2].set(config.market_demand*0.33, 1, id=name_of_buyers[2])

# ------------------------------------------------------------------------------------------------------------ #

for buyer in buyers:
    list_of_buyers.update({buyer.id: buyer})

player_total_demand = dict()

for item in list_of_buyers.keys():
    player_total_demand.update({item:list_of_buyers[item].total_demand})

for buyer in buyers:
    buyer.set_quantities(player_total_demand)

# PDA simulator

rounds = config.HOUR_AHEAD_AUCTIONS
cur_round = 0

while(cur_round < rounds):
    
    proximity = rounds - cur_round              # Proximity runs from 24 to 1 for a day-ahead PDA

    seller_cleared_quantity = dict()
    player_cleared_quantity = dict()

    for item in list_of_sellers.keys():
        seller_cleared_quantity.update({item:list_of_sellers[item].cleared_quantity})

    for item in list_of_buyers.keys():
        player_cleared_quantity.update({item:list_of_buyers[item].cleared_demand})

    for buyer in buyers:
        buyer.set_supply(seller_cleared_quantity)

    for buyer in buyers:
        buyer.set_demand(player_cleared_quantity)
    
    # asks dataframe
    asks_df = pd.DataFrame(list_of_sellers['cp_genco'].asks(), columns=['ID', 'Price', 'Quantity'])

    # occasionally generate small random asks
    if (proximity != config.HOUR_AHEAD_AUCTIONS) and (np.random.random() < 0.33):
        random_asks = pd.DataFrame([["miso", config.DEFAULT_MCP/10.0 + np.random.random()*0.7*config.DEFAULT_MCP, -np.random.normal(15, 1)]], columns=['ID', 'Price', 'Quantity'])
        asks_df = pd.concat([asks_df, random_asks], ignore_index=True)
        asks_df = asks_df.sort_values(by=['Price'])
    # print(asks_df)

    # bids dataframe
    if proximity == config.HOUR_AHEAD_AUCTIONS:
        bids_df = pd.DataFrame([["miso", -1e9, np.random.normal(800, 100)]], columns=['ID', 'Price', 'Quantity'])
    else:
        bids_df = pd.DataFrame(columns=['ID', 'Price', 'Quantity'])

    for buyer in list_of_buyers.keys():
        print("Original Requirement of ", list_of_buyers[buyer].id, ": ", (list_of_buyers[buyer].total_demand - list_of_buyers[buyer].cleared_demand))

    if config.parallel == True:       # debug this
        # parallelizing simulation using multiprocessing 
        manager = multiprocessing.Manager()
        return_buyers_df = manager.dict()

        jobs = []
        for i in range(len(list_of_buyers)):
            p = multiprocessing.Process(target=list_of_buyers[name_of_buyers[i]].bids, args=(rounds, cur_round, return_buyers_df))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()

        for i in range(len(list_of_buyers)):
            buyer_df = pd.DataFrame(return_buyers_df.values()[i], columns=['ID', 'Price', 'Quantity'])
            bids_df = pd.concat([bids_df,buyer_df], ignore_index=True)

        bids_df = bids_df.sort_values(by=['Price'])
        print(bids_df)

    else:
        for buyer in list_of_buyers.keys():
            buyer_df = pd.DataFrame(list_of_buyers[buyer].bids(rounds, cur_round), columns=['ID', 'Price', 'Quantity'])
            bids_df = pd.concat([bids_df,buyer_df], ignore_index=True)
        bids_df = bids_df.sort_values(by=['Price'])
        print(bids_df)
                
    # market clearing
    mcp, mcq, cleared_asks_df, cleared_bids_df, last_uncleared_ask = pda.clearing_mechanism(asks_df, bids_df)
    for buyer in buyers:
        if buyer.type == 'Continuous MCTS Regression':
            buyer.update_auction_data(proximity, mcp, mcq)
    
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
        
    if mcq != 0.0:
        for buyer in buyers:
            buyer.update_buy_limit_price_max(-last_uncleared_ask)
    
    print('\n----------From Handler: At Proxomity ', proximity, '------\n')
    print('MCP', mcp)
    print('MCQ', mcq)
    print()
    
    cur_round += 1