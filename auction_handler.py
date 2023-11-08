import gym
import math
import numpy as np
import pandas as pd

import genco
import pdauction
from broker import zi, zip, mcts_disc

from config import Config
from collections import OrderedDict

name_of_sellers = ['cp_genco']
name_of_buyers = ['ZI', 'MCTS_Dis']

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
# buyer3 = MCTS_Desc(config.market_demand*0.4, 5, id=name_of_buyers[2])

list_of_buyers.update({name_of_buyers[0]: buyer1})
list_of_buyers.update({name_of_buyers[1]: buyer2})
# list_of_buyers.update({name_of_buyers[2]: buyer3})

# PDA simulator

rounds = config.HOUR_AHEAD_AUCTIONS
cur_round = 0

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
        print("Original Requirement of ", list_of_buyers[buyer].id, ": ", (list_of_buyers[buyer].total_demand - list_of_buyers[buyer].cleared_demand))
        buyer_df = pd.DataFrame(list_of_buyers[buyer].bids(rounds, cur_round+1), columns=['ID', 'Price', 'Quantity'])
        bids_df = pd.concat([bids_df,buyer_df], ignore_index=True)

    bids_df = bids_df.sort_values(by=['Price'])

    print(bids_df)
                
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
    
    print('\n----------From Handler: At Proxomity ', proximity, '------\n')
    print('MCP', mcp)
    print('MCQ', mcq)
    print()
    
    cur_round += 1