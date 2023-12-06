import gym
import math
import numpy as np
import pandas as pd

import genco
import pdauction
from broker import zi, zip, mcts_disc, mcts_cont_spw

from config import Config
from collections import OrderedDict

name_of_sellers = ['cp_genco']
name_of_buyers = ['MCTS_Cont', 'ZI']

config = Config()
pda = gym.make('pdauction/Auctioneer-v0')

tot_cost_list = dict()
for item in name_of_buyers:
    tot_cost_list.update({item:0})

iterations = config.iter
for i in range(iterations):

    list_of_sellers = dict()
    list_of_buyers = dict()

    for seller in name_of_sellers:
        seller_obj = gym.make('genco/CPGenCo-v0')
        seller_obj.set_id('cp_genco')
        list_of_sellers.update({seller: seller_obj})
        
    buyer1 = gym.make('MCTS_Cont_SPW-v0')
    buyer1.set(config.market_demand*0.5, 1, id=name_of_buyers[0])
    buyer2 = gym.make('ZI-v0')
    buyer2.set(config.market_demand*0.5, 1, id=name_of_buyers[1])
    # buyer3 = MCTS_Desc(config.market_demand*0.4, 5, id=name_of_buyers[2])

    list_of_buyers.update({name_of_buyers[0]: buyer1})
    list_of_buyers.update({name_of_buyers[1]: buyer2})
    # list_of_buyers.update({name_of_buyers[2]: buyer3})

    player_total_demand = dict()
    for item in list_of_buyers.keys():
        player_total_demand.update({item:list_of_buyers[item].total_demand})

    buyer1.set_quantities(player_total_demand)

    # PDA simulator

    rounds = config.HOUR_AHEAD_AUCTIONS
    cur_round = 0
    cost_list = dict()
    for item in list_of_buyers.keys():
        cost_list.update({item:0})

    while(cur_round < rounds):
        
        proximity = rounds - cur_round              # Proximity runs from 24 to 1 for a day-ahead PDA

        seller_cleared_quantity = dict()
        player_cleared_quantity = dict()

        for item in list_of_sellers.keys():
            seller_cleared_quantity.update({item:list_of_sellers[item].cleared_quantity})

        for item in list_of_buyers.keys():
            player_cleared_quantity.update({item:list_of_buyers[item].cleared_demand})

        buyer1.set_supply(seller_cleared_quantity)
        buyer1.set_demand(player_cleared_quantity)
        
        # asks dataframe
        asks_df = pd.DataFrame(list_of_sellers['cp_genco'].asks(), columns=['ID', 'Price', 'Quantity'])

        # bids dataframe
        if proximity == config.HOUR_AHEAD_AUCTIONS:
            bids_df = pd.DataFrame([["miso", -1e9, np.random.normal(800, 100)]], columns=['ID', 'Price', 'Quantity'])
        else:
            bids_df = pd.DataFrame(columns=['ID', 'Price', 'Quantity'])

        # for buyer in list_of_buyers.keys():
        #     print("Original Requirement of ", list_of_buyers[buyer].id, ": ", (list_of_buyers[buyer].total_demand - list_of_buyers[buyer].cleared_demand))

        for buyer in list_of_buyers.keys():
            buyer_df = pd.DataFrame(list_of_buyers[buyer].bids(rounds, cur_round), columns=['ID', 'Price', 'Quantity'])
            bids_df = pd.concat([bids_df,buyer_df], ignore_index=True)

        bids_df = bids_df.sort_values(by=['Price'])

        # print(bids_df)
                    
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
                cost_list[buyer] += mcp*buyer_cq 
        
        # print('\n----------From Handler: At Proxomity ', proximity, '------\n')
        # print('MCP', mcp)
        # print('MCQ', mcq)
        # print()
        
        cur_round += 1

    # TO DO: If brokers are placing matket order at the last proximity for total clearing

    for item in name_of_buyers:
        tot_cost_list[item] += cost_list[item]

    print('Individual Purchase Cost:')
    for item in name_of_buyers:
        print(item, ': ', cost_list[item])

print('Average Purchase Cost:')
for item in name_of_buyers:
    print(item, ': ', tot_cost_list[item]/len(tot_cost_list))
