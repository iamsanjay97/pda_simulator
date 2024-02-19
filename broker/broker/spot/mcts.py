import io
import sys
import math
import random
import time

class MCTS:

    def __init__(self):
        self.root = None
        self.total_successful_bids = 0.0
        self.lastmcts_clearing_price = 0.0
        self.last_price_diff_per_trade_action = 0.0
        self.actionset_type = 3.0
        self.vol_modi1 = 0.8
        self.vol_modi2 = 1.0
        self.vol_modi3 = 1.2
        self.arr_mcts_pred_clearing_price = [0.0] * 25
        self.threshold_low_auctions_perc = 40.0
        self.boo_nobid = False
        self.actions = []
        self.player_name = "MCTS"
        self.STDDEV2015 = 21.0
        self.MEANSTDDEV2015 = 6.5
        self.root = TreeNode()
        print('MCTS Initiated')

    def setup(self):
        self.actionset_type = 2.0
        self.vol_modi1 = 0.9
        self.vol_modi2 = 1.0
        self.vol_modi3 = 0.0

        if self.actionset_type == 1.0:
            action = Action("0", -2.0, 2.0, False, ACTION_TYPE.BUY, self.vol_modi1)
            self.actions.append(action)
            action = Action("1", -2.0, 1.0, False, ACTION_TYPE.BUY, self.vol_modi1)
            self.actions.append(action)
            action = Action("2", -2.0, 0.0, False, ACTION_TYPE.BUY, self.vol_modi1)
            self.actions.append(action)
            action = Action("3", -2.0, -1.0, False, ACTION_TYPE.BUY, self.vol_modi1)
            self.actions.append(action)
            action = Action("4", -1.0, 2.0, False, ACTION_TYPE.BUY, self.vol_modi1)
            self.actions.append(action)
            action = Action("5", -1.0, 1.0, False, ACTION_TYPE.BUY, self.vol_modi1)
            self.actions.append(action)
            action = Action("6", -1.0, 0.0, False, ACTION_TYPE.BUY, self.vol_modi1)
            self.actions.append(action)
            action = Action("7", 0.0, 2.0, False, ACTION_TYPE.BUY, self.vol_modi1)
            self.actions.append(action)
            action = Action("8", 0.0, 1.0, False, ACTION_TYPE.BUY, self.vol_modi1)
            self.actions.append(action)
            action = Action("9", 1.0, 2.0, False, ACTION_TYPE.BUY, self.vol_modi1)
            self.actions.append(action)
            action = Action("10", 0.0, 0.0, True, ACTION_TYPE.NO_BID, 1.0)
            self.actions.append(action)
            action = Action("11", -2.0, 2.0, False, ACTION_TYPE.BUY, self.vol_modi2)
            self.actions.append(action)
            action = Action("12", -2.0, 1.0, False, ACTION_TYPE.BUY, self.vol_modi2)
            self.actions.append(action)
            action = Action("13", -2.0, 0.0, False, ACTION_TYPE.BUY, self.vol_modi2)
            self.actions.append(action)
            action = Action("14", -2.0, -1.0, False, ACTION_TYPE.BUY, self.vol_modi2)
            self.actions.append(action)
            action = Action("15", -1.0, 2.0, False, ACTION_TYPE.BUY, self.vol_modi2)
            self.actions.append(action)
            action = Action("16", -1.0, 1.0, False, ACTION_TYPE.BUY, self.vol_modi2)
            self.actions.append(action)
            action = Action("17", -1.0, 0.0, False, ACTION_TYPE.BUY, self.vol_modi2)
            self.actions.append(action)
            action = Action("18", 0.0, 2.0, False, ACTION_TYPE.BUY, self.vol_modi2)
            self.actions.append(action)
            action = Action("19", 0.0, 1.0, False, ACTION_TYPE.BUY, self.vol_modi2)
            self.actions.append(action)
            action = Action("20", 1.0, 2.0, False, ACTION_TYPE.BUY, self.vol_modi2)
            self.actions.append(action)
            action = Action("21", -2.0, 2.0, False, ACTION_TYPE.SELL, self.vol_modi3)
            self.actions.append(action)
            action = Action("22", -2.0, 1.0, False, ACTION_TYPE.SELL, self.vol_modi3)
            self.actions.append(action)
            action = Action("23", -2.0, 0.0, False, ACTION_TYPE.SELL, self.vol_modi3)
            self.actions.append(action)
            action = Action("24", -2.0, -1.0, False, ACTION_TYPE.SELL, self.vol_modi3)
            self.actions.append(action)
            action = Action("25", -1.0, 2.0, False, ACTION_TYPE.SELL, self.vol_modi3)
            self.actions.append(action)
            action = Action("26", -1.0, 1.0, False, ACTION_TYPE.SELL, self.vol_modi3)
            self.actions.append(action)
            action = Action("27", -1.0, 0.0, False, ACTION_TYPE.SELL, self.vol_modi3)
            self.actions.append(action)
            action = Action("28", 0.0, 2.0, False, ACTION_TYPE.SELL, self.vol_modi3)
            self.actions.append(action)
            action = Action("29", 0.0, 1.0, False, ACTION_TYPE.SELL, self.vol_modi3)
            self.actions.append(action)
            action = Action("30", 1.0, 2.0, False, ACTION_TYPE.SELL, self.vol_modi3)
            self.actions.append(action)
        elif self.actionset_type == 2.0:
            action = Action("0", -2.0, 2.0, False, ACTION_TYPE.BUY, self.vol_modi1)
            self.actions.append(action)
            action = Action("1", -2.0, 1.0, False, ACTION_TYPE.BUY, self.vol_modi1)
            self.actions.append(action)
            action = Action("2", -2.0, 0.0, False, ACTION_TYPE.BUY, self.vol_modi1)
            self.actions.append(action)
            action = Action("3", -2.0, -1.0, False, ACTION_TYPE.BUY, self.vol_modi1)
            self.actions.append(action)
            action = Action("4", -1.0, 2.0, False, ACTION_TYPE.BUY, self.vol_modi1)
            self.actions.append(action)
            action = Action("5", -1.0, 1.0, False, ACTION_TYPE.BUY, self.vol_modi1)
            self.actions.append(action)
            action = Action("6", -1.0, 0.0, False, ACTION_TYPE.BUY, self.vol_modi1)
            self.actions.append(action)
            action = Action("7", 0.0, 2.0, False, ACTION_TYPE.BUY, self.vol_modi1)
            self.actions.append(action)
            action = Action("8", 0.0, 1.0, False, ACTION_TYPE.BUY, self.vol_modi1)
            self.actions.append(action)
            action = Action("9", 1.0, 2.0, False, ACTION_TYPE.BUY, self.vol_modi1)
            self.actions.append(action)
            action = Action("10", 0.0, 0.0, True, ACTION_TYPE.NO_BID, 1.0)
            self.actions.append(action)
            action = Action("11", -2.0, 2.0, False, ACTION_TYPE.BUY, self.vol_modi2)
            self.actions.append(action)
            action = Action("12", -2.0, 1.0, False, ACTION_TYPE.BUY, self.vol_modi2)
            self.actions.append(action)
            action = Action("13", -2.0, 0.0, False, ACTION_TYPE.BUY, self.vol_modi2)
            self.actions.append(action)
            action = Action("14", -2.0, -1.0, False, ACTION_TYPE.BUY, self.vol_modi2)
            self.actions.append(action)
            action = Action("15", -1.0, 2.0, False, ACTION_TYPE.BUY, self.vol_modi2)
            self.actions.append(action)
            action = Action("16", -1.0, 1.0, False, ACTION_TYPE.BUY, self.vol_modi2)
            self.actions.append(action)
            action = Action("17", -1.0, 0.0, False, ACTION_TYPE.BUY, self.vol_modi2)
            self.actions.append(action)
            action = Action("18", 0.0, 2.0, False, ACTION_TYPE.BUY, self.vol_modi2)
            self.actions.append(action)
            action = Action("19", 0.0, 1.0, False, ACTION_TYPE.BUY, self.vol_modi2)
            self.actions.append(action)
            action = Action("20", 1.0, 2.0, False, ACTION_TYPE.BUY, self.vol_modi2)
            self.actions.append(action)
        elif self.actionset_type == 3.0:
            action = Action("0", -2.0, 2.0, False, ACTION_TYPE.BUY, self.vol_modi1)
            self.actions.append(action)
            action = Action("1", -2.0, 1.0, False, ACTION_TYPE.BUY, self.vol_modi1)
            self.actions.append(action)
            action = Action("2", -2.0, 0.0, False, ACTION_TYPE.BUY, self.vol_modi1)
            self.actions.append(action)
            action = Action("3", -2.0, -1.0, False, ACTION_TYPE.BUY, self.vol_modi1)
            self.actions.append(action)
            action = Action("4", -1.0, 2.0, False, ACTION_TYPE.BUY, self.vol_modi1)
            self.actions.append(action)
            action = Action("5", -1.0, 1.0, False, ACTION_TYPE.BUY, self.vol_modi1)
            self.actions.append(action)
            action = Action("6", -1.0, 0.0, False, ACTION_TYPE.BUY, self.vol_modi1)
            self.actions.append(action)
            action = Action("7", 0.0, 2.0, False, ACTION_TYPE.BUY, self.vol_modi1)
            self.actions.append(action)
            action = Action("8", 0.0, 1.0, False, ACTION_TYPE.BUY, self.vol_modi1)
            self.actions.append(action)
            action = Action("9", 1.0, 2.0, False, ACTION_TYPE.BUY, self.vol_modi1)
            self.actions.append(action)
            action = Action("10", 0.0, 0.0, True, ACTION_TYPE.NO_BID, 1.0)
            self.actions.append(action)
            action = Action("11", -2.0, 2.0, False, ACTION_TYPE.BUY, self.vol_modi2)
            self.actions.append(action)
            action = Action("12", -2.0, 1.0, False, ACTION_TYPE.BUY, self.vol_modi2)
            self.actions.append(action)
            action = Action("13", -2.0, 0.0, False, ACTION_TYPE.BUY, self.vol_modi2)
            self.actions.append(action)
            action = Action("14", -2.0, -1.0, False, ACTION_TYPE.BUY, self.vol_modi2)
            self.actions.append(action)
            action = Action("15", -1.0, 2.0, False, ACTION_TYPE.BUY, self.vol_modi2)
            self.actions.append(action)
            action = Action("16", -1.0, 1.0, False, ACTION_TYPE.BUY, self.vol_modi2)
            self.actions.append(action)
            action = Action("17", -1.0, 0.0, False, ACTION_TYPE.BUY, self.vol_modi2)
            self.actions.append(action)
            action = Action("18", 0.0, 2.0, False, ACTION_TYPE.BUY, self.vol_modi2)
            self.actions.append(action)
            action = Action("19", 0.0, 1.0, False, ACTION_TYPE.BUY, self.vol_modi2)
            self.actions.append(action)
            action = Action("20", 1.0, 2.0, False, ACTION_TYPE.BUY, self.vol_modi2)
            self.actions.append(action)
            action = Action("21", -2.0, 2.0, False, ACTION_TYPE.BUY, self.vol_modi3)
            self.actions.append(action)
            action = Action("22", -2.0, 1.0, False, ACTION_TYPE.BUY, self.vol_modi3)
            self.actions.append(action)
            action = Action("23", -2.0, 0.0, False, ACTION_TYPE.BUY, self.vol_modi3)
            self.actions.append(action)
            action = Action("24", -2.0, -1.0, False, ACTION_TYPE.BUY, self.vol_modi3)
            self.actions.append(action)
            action = Action("25", -1.0, 2.0, False, ACTION_TYPE.BUY, self.vol_modi3)
            self.actions.append(action)
            action = Action("26", -1.0, 1.0, False, ACTION_TYPE.BUY, self.vol_modi3)
            self.actions.append(action)
            action = Action("27", -1.0, 0.0, False, ACTION_TYPE.BUY, self.vol_modi3)
            self.actions.append(action)
            action = Action("28", 0.0, 2.0, False, ACTION_TYPE.BUY, self.vol_modi3)
            self.actions.append(action)
            action = Action("29", 0.0, 1.0, False, ACTION_TYPE.BUY, self.vol_modi3)
            self.actions.append(action)
            action = Action("30", 1.0, 2.0, False, ACTION_TYPE.BUY, self.vol_modi3)
            self.actions.append(action)
        else:
            action = Action("0", -2.0, 2.0, False, ACTION_TYPE.BUY, self.vol_modi1)
            self.actions.append(action)
            action = Action("1", -2.0, 1.0, False, ACTION_TYPE.BUY, self.vol_modi1)
            self.actions.append(action)
            action = Action("2", -2.0, 0.0, False, ACTION_TYPE.BUY, self.vol_modi1)
            self.actions.append(action)
            action = Action("3", -2.0, -1.0, False, ACTION_TYPE.BUY, self.vol_modi1)
            self.actions.append(action)
            action = Action("4", -1.0, 2.0, False, ACTION_TYPE.BUY, self.vol_modi1)
            self.actions.append(action)
            action = Action("5", -1.0, 1.0, False, ACTION_TYPE.BUY, self.vol_modi1)
            self.actions.append(action)
            action = Action("6", -1.0, 0.0, False, ACTION_TYPE.BUY, self.vol_modi1)
            self.actions.append(action)
            action = Action("7", 0.0, 2.0, False, ACTION_TYPE.BUY, self.vol_modi1)
            self.actions.append(action)
            action = Action("8", 0.0, 1.0, False, ACTION_TYPE.BUY, self.vol_modi1)
            self.actions.append(action)
            action = Action("9", 1.0, 2.0, False, ACTION_TYPE.BUY, self.vol_modi1)
            self.actions.append(action)
            action = Action("10", 0.0, 0.0, True, ACTION_TYPE.NO_BID, 1.0)
            self.actions.append(action)
            action = Action("11", -2.0, 2.0, False, ACTION_TYPE.BUY, self.vol_modi2)
            self.actions.append(action)
            action = Action("12", -2.0, 1.0, False, ACTION_TYPE.BUY, self.vol_modi2)
            self.actions.append(action)
            action = Action("13", -2.0, 0.0, False, ACTION_TYPE.BUY, self.vol_modi2)
            self.actions.append(action)
            action = Action("14", -2.0, -1.0, False, ACTION_TYPE.BUY, self.vol_modi2)
            self.actions.append(action)
            action = Action("15", -1.0, 2.0, False, ACTION_TYPE.BUY, self.vol_modi2)
            self.actions.append(action)
            action = Action("16", -1.0, 1.0, False, ACTION_TYPE.BUY, self.vol_modi2)
            self.actions.append(action)
            action = Action("17", -1.0, 0.0, False, ACTION_TYPE.BUY, self.vol_modi2)
            self.actions.append(action)
            action = Action("18", 0.0, 2.0, False, ACTION_TYPE.BUY, self.vol_modi2)
            self.actions.append(action)
            action = Action("19", 0.0, 1.0, False, ACTION_TYPE.BUY, self.vol_modi2)
            self.actions.append(action)
            action = Action("20", 1.0, 2.0, False, ACTION_TYPE.BUY, self.vol_modi2)
            self.actions.append(action)
            action = Action("21", -2.0, 2.0, False, ACTION_TYPE.BUY, self.vol_modi3)
            self.actions.append(action)
            action = Action("22", -2.0, 1.0, False, ACTION_TYPE.BUY, self.vol_modi3)
            self.actions.append(action)
            action = Action("23", -2.0, 0.0, False, ACTION_TYPE.BUY, self.vol_modi3)
            self.actions.append(action)
            action = Action("24", -2.0, -1.0, False, ACTION_TYPE.BUY, self.vol_modi3)
            self.actions.append(action)
            action = Action("25", -1.0, 2.0, False, ACTION_TYPE.BUY, self.vol_modi3)
            self.actions.append(action)
            action = Action("26", -1.0, 1.0, False, ACTION_TYPE.BUY, self.vol_modi3)
            self.actions.append(action)
            action = Action("27", -1.0, 0.0, False, ACTION_TYPE.BUY, self.vol_modi3)
            self.actions.append(action)
            action = Action("28", 0.0, 2.0, False, ACTION_TYPE.BUY, self.vol_modi3)
            self.actions.append(action)
            action = Action("29", 0.0, 1.0, False, ACTION_TYPE.BUY, self.vol_modi3)
            self.actions.append(action)
            action = Action("30", 1.0, 2.0, False, ACTION_TYPE.BUY, self.vol_modi3)
            self.actions.append(action)
            print("self SHOULD NEVER HAPPEN! \nNO ACTIONSET TYPE SELECTED FOR MCTS\nDEFAULT SET LOADED 30BUY")


    def get_best_mcts_move(self, observer):
        arr_counter_higher_bids = [None] * 25
        self.root.hour_ahead_auction = observer.hour_ahead + 1
        for j in range(observer.hour_ahead + 1):
            self.arr_mcts_pred_clearing_price[j] = observer.arr_clearing_prices[j]
            arr_counter_higher_bids[j] = 0

        self.root.minmcts_clearing_price = self.arr_mcts_pred_clearing_price[observer.hour_ahead]
        start_time = time.time()
        i = 0

        while True:
            self.root.runMonteCarlo(self.actions, self, observer)
            end_time = time.time()
            elapsed_time = end_time - start_time
            if elapsed_time > 0.125:
                break
            i += 1
        
        return self.root.final_select(observer)
    

    def rand_error(self, error_rate):
        min = 100 - error_rate
        max = 100 + error_rate
        rand = random.Random()
        random_num = rand.nextInt(max - min + 1) + min
        random_gauss = rand.nextGaussian() * error_rate + 0.0
        random_gauss = 100.0 + random_gauss
        
        return random_gauss / 100.0
    

class TreeNode:

    r = random.Random()
    epsilon = 1e-6
    
    # n_visit, hour_ahead_auction

    def __init__(self, nobid: bool):
        self.nobid = nobid
        print('Treenode Initiated ... nobid')

    def __init__(self):
        self.n_visits = 0
        self.tot_value = 0
        self.current_node_cost_avg = 0
        self.current_node_cost_last = 0
        self.minmcts_clearing_price = 0
        self.maxmcts_clearing_price = 0
        self.min_mult = 0
        self.max_mult = 0
        self.hour_ahead_auction = 24
        self.applied_action = 0
        self.nobid = True
        self.action_type = None
        self.vol_percentage = 0
        self.action_name = None
        self.balancing_price = 200
        self.children = []  # List for children nodes
        print('Treenode Initiated ... default')

    def set(self, tn):
        node = self.TreeNode()
        node.n_visits = tn.n_visits
        node.tot_value = tn.tot_value
        node.current_node_cost_avg = tn.current_node_cost_avg
        node.current_node_cost_last = tn.current_node_cost_last
        node.minmcts_clearing_price = tn.minmcts_clearing_price
        node.maxmcts_clearing_price = tn.maxmcts_clearing_price
        node.min_mult = tn.min_mult
        node.max_mult = tn.max_mult
        node.hour_ahead_auction = tn.hour_ahead_auction
        node.applied_action = tn.applied_action
        node.nobid = tn.nobid
        node.action_type = tn.action_type
        node.vol_percentage = tn.vol_percentage
        node.action_name = tn.action_name
        node.balancing_price = 200
        node.children = []  # List for children nodes
        return node

    
    def unvisited_children(self, tn):
        count = 0
        for child in tn.children:
            if child.n_visits == 0:
                count += 1
        return count
    

    def run_monte_carlo(self, actions, mcts, ob):
        sim_cost = 0.0
        needed_energy = ob.needed_energy_per_broker 
        ini_needed_energy = ob.initial_needed_energy_mcts_broker 
        visited = []
        cur = self

        visited.append(self)
        if cur.is_leaf():
            print("cur.is_leaf() is 0! This should not happen!")

        while not cur.is_leaf():
            if cur.children is None:
                cur.expand(actions, mcts, ob)

            unvisited_children = unvisited_children(cur) 
            if unvisited_children == len(actions):
                cur = cur.select_random(mcts, ob)
                visited.append(cur)

                ret_value = self.rollout(cur, mcts.arr_mcts_pred_clearing_price, ob, needed_energy, ini_needed_energy, actions, mcts) 
                needed_energy -= ret_value[0] 
                sim_cost += -1.0 * ret_value[1]
                break

            cur = cur.select(mcts, ob) 

            ret_value = self.simulation(cur, mcts.arr_mcts_pred_clearing_price, ob, needed_energy, ini_needed_energy) 
            needed_energy -= ret_value[0] 
            sim_cost += -1.0 * ret_value[1]

            visited.append(cur)

        # bprice = ob.market_manager.get_balancing_price(-1.0 * needed_energy)
        bprice = self.balancing_price

        balancing_sim_cost = abs(needed_energy) * bprice
        sim_cost += balancing_sim_cost

        for node in visited:
            node.update_stats(sim_cost, balancing_sim_cost) 

        
    def select_random(self, mcts, observer):
        r = random.Random()
        i = r.nextInt(mcts.actions.size())
        return self.children[i]
    

    def expand(self, actions, mcts, ob):
        n_actions = len(actions)
        self.children = [None] * n_actions 
        new_hour_ahead_auction = self.hour_ahead_auction - 1
        for i in range(n_actions):
            print('Before')
            self.children[i] = self.TreeNode()
            print('After')
            self.children[i].hour_ahead_auction = new_hour_ahead_auction

            mean = ob.arr_clearing_prices[new_hour_ahead_auction] 
            stddev = ob.STDDEV 
            prices = actions[i].get_adjusted_price(mean, stddev)
            self.children[i].min_mcts_clearing_price = prices[0]
            self.children[i].max_mcts_clearing_price = prices[1]
            self.children[i].applied_action = i
            self.children[i].nobid = actions[i].nobid
            self.children[i].max_mult = actions[i].max_mult
            self.children[i].min_mult = actions[i].min_mult
            self.children[i].action_type = actions[i].type
            self.children[i].vol_percentage = actions[i].percentage
            self.children[i].current_node_cost_avg = 0.0
            self.children[i].current_node_cost_last = 0.0
            self.children[i].action_name = actions[i].action_name

    def final_select(self, ob):
        print_on = False
        selected = None
        best_value = float("-inf") 

        try:
            for child in self.children:
                totl_point = child.tot_value

                # dividend = ob.market_manager.get_balancing_price(-1.0 * ob.initial_needed_energy_mcts_broker) * abs(ob.initial_needed_energy_mcts_broker)  # Replace with actual attribute access
                dividend = self.balancing_price
                temp_t = totl_point
                totl_point = 1.0 - totl_point / dividend

                visit_point = math.sqrt(2.0 * math.log(self.n_visits + 1.0) / (child.n_visits + self.epsilon))

                rand_point = self.r.random() * self.epsilon
                uct_value = totl_point + visit_point + rand_point
                if print_on:
                    print(f"Action {child.applied_action} UCT value = {uct_value} totlPoint {totl_point} nvisitPoint {visit_point} c.nvisits {child.n_visits} totalVisits {self.n_visits}")
                if uct_value > best_value:
                    selected = child
                    best_value = uct_value
                    if print_on:
                        print(" [best] ")
                elif print_on:
                    print("")
        except Exception as e:
            print(e)

        return selected
    

    def select(self, mcts, ob):
        print_on = False
        selected = None
        best_value = float("-inf")

        count_lower_auction = 0
        boo_nobid = False
        for jj in range(self.hour_ahead_auction - 1, -1, -1):
            if ob.arr_clearing_prices[self.hour_ahead_auction - 1] > ob.arr_clearing_prices[jj]:
                count_lower_auction += 1
        threshold_limit = (self.hour_ahead_auction + 1) * (mcts.threshold_low_auctions_perc / 100.0)
        if count_lower_auction > threshold_limit:
            boo_nobid = False

        for child in self.children:
            if boo_nobid:
                if child.nobid:
                    selected = child
            else:
                n_visit_value = child.n_visits if child.n_visits > 0 else 1.0 + self.epsilon
                totl_point = child.tot_value

                # dividend = ob.market_manager.get_balancing_price(-1.0 * ob.initial_needed_energy_mcts_broker) * abs(ob.initial_needed_energy_mcts_broker)
                dividend = self.balancing_price
                totl_point = 1.0 - totl_point / dividend

                visit_point = math.sqrt(2.0 * math.log(self.n_visits + 1.0) / n_visit_value)

                rand_point = self.r.random() * self.epsilon
                uct_value = totl_point + visit_point + rand_point
                if print_on:
                    print(f"Action {child.applied_action} UCT value = {uct_value} totlPoint {totl_point} nvisitPoint {visit_point} c.nvisits {child.n_visits} totalVisits {self.n_visits}")
                if uct_value > best_value:
                    selected = child
                    best_value = uct_value

        if print_on:
            print(f"HourAhead {selected.hour_ahead_auction} Action {selected.applied_action}")
            print()
        return selected


    def is_leaf(self) -> bool:
        return self.hour_ahead_auction == 0
    

    def rollout(self, temp_node, arr_pred_clearing_price, ob, needed_mwh, ini_needed_energy, actions, mcts):
        print('Inside rollout')
        tn = self.set(temp_node)  # Create a copy of the node

        total_bid_volume = 0.0
        cost_value = 0.0
        while needed_mwh != 0.0:
            single_bid_volume = 0.0
            if not tn.nobid:
                number_of_bids = 10.0
                unit_price_increment = 1.0
                clearing_price = abs(random.gauss(0, ob.STDDEV) + arr_pred_clearing_price[tn.hour_ahead_auction])
                limit_price = tn.min_mcts_clearing_price
                max_price = tn.min_mcts_clearing_price
                price_range = tn.max_mcts_clearing_price - tn.min_mcts_clearing_price
                min_mwh = 1.0

                unit_price_increment = price_range / number_of_bids
                if tn.action_type == ACTION_TYPE.BUY:
                    surplus = abs(ob.initial_needed_energy_mcts_broker) * (tn.vol_percentage - 1.0)
                    total_e = surplus + abs(needed_mwh)
                    min_mwh = abs(total_e) / number_of_bids
                    for i in range(1, int(number_of_bids) + 1):
                        if limit_price >= clearing_price:
                            cost_value += min_mwh * clearing_price
                            total_bid_volume += min_mwh
                            single_bid_volume += min_mwh
                        limit_price += unit_price_increment
                else:
                    surplus = - needed_mwh
                    min_mwh = abs(ini_needed_energy) * (1.0 - tn.vol_percentage)
                    min_mwh = max(min_mwh, surplus)
                    # bprice = abs(ob.market_manager.get_balancing_price(min_mwh))  # Replace with actual method call
                    bprice = self.balancing_price
                    min_mwh /= number_of_bids
                    for i in range(1, number_of_bids + 1):
                        if bprice <= clearing_price:
                            cost_value -= min_mwh * clearing_price
                            total_bid_volume -= min_mwh
                            single_bid_volume -= min_mwh
                        bprice += unit_price_increment

                if tn.hour_ahead_auction == ob.hour_ahead:
                    tn.current_node_cost_last = -1.0 * single_bid_volume * clearing_price

            needed_mwh -= single_bid_volume
            if tn.hour_ahead_auction == 0:
                break
            tn.expand(actions, mcts, ob)
            tn = tn.select_random(mcts, ob)  # Replace with actual method name

        return total_bid_volume, cost_value
    

    def simulation(self, arr_pred_clearing_price, ob, needed_mwh, ini_needed_energy):
        total_bid_volume = 0.0
        cost_value = 0.0

        if not self.nobid:
            number_of_bids = 10.0
            unit_price_increment = 1.0
            clearing_price = abs(random.gauss(0, ob.STDDEV) + arr_pred_clearing_price[self.hour_ahead_auction])
            limit_price = self.min_mcts_clearing_price
            max_price = self.max_mcts_clearing_price
            price_range = max_price - limit_price
            min_mwh = 1.0

            unit_price_increment = price_range / number_of_bids
            if self.action_type == ACTION_TYPE.BUY: 
                surplus = abs(ob.initial_needed_energy_mcts_broker) * (self.vol_percentage - 1.0)
                total_e = surplus + abs(needed_mwh)
                min_mwh = abs(total_e) / number_of_bids
                for i in range(1, int(number_of_bids) + 1):
                    if limit_price >= clearing_price:
                        cost_value += min_mwh * clearing_price
                        total_bid_volume += min_mwh
                    limit_price += unit_price_increment
            else:
                surplus = - needed_mwh
                min_mwh = abs(ini_needed_energy) * (1.0 - self.vol_percentage)
                min_mwh = max(min_mwh, surplus)
                # bprice = abs(ob.market_manager.get_balancing_price(min_mwh)) 
                bprice = self.balancing_price
                min_mwh /= number_of_bids
                for i in range(1, int(number_of_bids) + 1):
                    if bprice <= clearing_price:
                        cost_value -= min_mwh * clearing_price
                        total_bid_volume -= min_mwh
                    bprice += unit_price_increment

            if self.hour_ahead_auction == ob.hour_ahead:
                self.current_node_cost_last = -1.0 * total_bid_volume * clearing_price

        return total_bid_volume, cost_value


    def update_stats(self, sim_cost, balancing_sim_cost) -> None:
        self.tot_value = ((self.tot_value * self.n_visits + sim_cost) / (self.n_visits + 1.0))
        self.current_node_cost_avg = ((self.current_node_cost_avg * self.n_visits + self.current_node_cost_last) / (self.n_visits + 1.0))
        self.n_visits += 1.0


    def arity(self) -> int:
        return 0 if self.children is None else len(self.children)  # Pythonic way to check for empty list


    def get_children(self, hour_ahead, count):
        if count == hour_ahead:
            return self.children
        return self.get_children(hour_ahead, count + 1)
    

class Action:
    """Represents an action with adjustable price calculation."""

    def __init__(self, action_name, min_mult, max_mult, no_bid, action_type, percentage):
        """Initializes the Action object."""
        self.action_name = action_name
        self.min_mult = min_mult
        self.max_mult = max_mult
        self.no_bid = no_bid
        self.type = action_type
        self.percentage = percentage

    def get_adjusted_price(self, mean_price, stddev):
        """Calculates and returns the adjusted price range."""
        min_price = mean_price + self.min_mult * stddev
        max_price = mean_price + self.max_mult * stddev
        if min_price > max_price:
            min_price, max_price = max_price, min_price
        return min_price, max_price

    def __str__(self) -> str:
        """Returns a string representation of the Action object."""
        return f"[action: {self.action_name} minMult: {self.min_mult} maxMult: {self.max_mult} nobid: {self.no_bid}]"


class ACTION_TYPE:
    """Enum for action types."""

    BUY = "BUY"
    SELL = "SELL"
    NO_BID = "NO_BID"


class Observer:
    
    def __init__(self):
        self.day = 0
        self.hour = 0
        self.hour_ahead = 0
        self.current_time_slot = 0
        self.moving_avg_err_bal = 0.0
        self.MCTSSimulation = 5000
        self.HOUR_AHEAD_AUCTIONS = 24
        self.total_hours_ahead = 0
        self.arr_clearing_prices = [0.0] * self.HOUR_AHEAD_AUCTIONS
        self.needed_energy_per_broker = 0.0
        self.initial_needed_energy_mcts_broker = 0.0
        self.STDDEV2016 = 13.38
        self.MEANSTDDEV2016 = -0.24
        self.STDDEV2017 = 17.68961
        self.MEAN2017 = 39.50736
        self.STDDEV = self.STDDEV2016

    def set_time(self, day, hour, hour_ahead, current_time_slot):
        self.day = day
        self.hour = hour
        self.hour_ahead = hour_ahead
        self.current_time_slot = current_time_slot

    def print(self):
        print("day", self.day, "hour", self.hour, "hourAhead", self.hour_ahead, "currentTimeSlot", self.current_time_slot)