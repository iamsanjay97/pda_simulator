import sys
import math
import random

sys.path.append('/home/sanjay/Research/MCTS/Codes/pda_simulator/broker/broker/spot')
from mcts import MCTS
from action import Action
from observer import Observer

class TreeNode(object):

    r = random.Random()
    epsilon = 1e-6
    children = []  # List for children nodes

    def __init__(self, nobid: bool):
        self.nobid = nobid

    def __init__(self, tn):
        self.n_visits = tn.n_visits
        self.tot_value = tn.tot_value
        self.current_node_cost_avg = tn.current_node_cost_avg
        self.current_node_cost_last = tn.current_node_cost_last
        self.minmcts_clearing_price = tn.minmcts_clearing_price
        self.maxmcts_clearing_price = tn.maxmcts_clearing_price
        self.min_mult = tn.min_mult
        self.max_mult = tn.max_mult
        self.hour_ahead_auction = tn.hour_ahead_auction
        self.applied_action = tn.applied_action
        self.nobid = tn.nobid
        self.action_type = tn.action_type
        self.vol_percentage = tn.vol_percentage
        self.action_name = tn.action_name

    
    def unvisited_children(self, tn):
        count = 0
        for child in tn.children:
            if child.n_visits == 0:
                count += 1
        return count
    

    def run_monte_carlo(self, actions: list[Action], mcts: MCTS, ob: Observer):
        sim_cost = 0.0
        needed_energy = ob.needed_energy_per_broker 
        ini_needed_energy = ob.initial_needed_energy_mcts_broker 
        visited: list[TreeNode] = []
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

        bprice = ob.market_manager.get_balancing_price(-1.0 * needed_energy)

        balancing_sim_cost = abs(needed_energy) * bprice
        sim_cost += balancing_sim_cost

        for node in visited:
            node.update_stats(sim_cost, balancing_sim_cost) 

        
    def select_random(self, mcts, observer):
        r = random.Random()
        i = r.nextInt(mcts.actions.size())
        return self.children[i]
    

    def expand(self, actions: list[Action], mcts: MCTS, ob: Observer):
        n_actions = len(actions)
        self.children = [None] * n_actions 
        new_hour_ahead_auction = self.hour_ahead_auction - 1
        for i in range(n_actions):
            self.children[i] = TreeNode()
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

    def final_select(self, ob: Observer):
        print_on = False
        selected = None
        best_value = float("-inf") 

        try:
            for child in self.children:
                totl_point = child.tot_value

                dividend = ob.market_manager.get_balancing_price(-1.0 * ob.initial_needed_energy_mcts_broker) * abs(ob.initial_needed_energy_mcts_broker)  # Replace with actual attribute access
                temp_t = totl_point
                totl_point = 1.0 - totl_point / dividend

                visit_point = math.sqrt(2.0 * math.log(self.n_visits + 1.0) / (child.n_visits + TreeNode.epsilon))

                rand_point = TreeNode.r.random() * TreeNode.epsilon
                uct_value = totl_point + visit_point + rand_point
                if print_on:
                    print(f"Action {child.applied_action} UCT value = {uct_value} totlPoint {totl_point} nvisitPoint {visit_point} c.nvisits {child.n_visits} totalVisits {self.n_visits}")
                if totl_point > best_value:
                    selected = child
                    best_value = totl_point
                    if print_on:
                        print(" [best] ")
                elif print_on:
                    print("")
        except Exception as ex:
            ex.printStackTrace()

        return selected
    

    def select(self, mcts: MCTS, ob: Observer):
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
                n_visit_value = child.n_visits if child.n_visits > 0 else 1.0 + TreeNode.epsilon
                totl_point = child.tot_value

                dividend = ob.market_manager.get_balancing_price(-1.0 * ob.initial_needed_energy_mcts_broker) * abs(ob.initial_needed_energy_mcts_broker)
                totl_point = 1.0 - totl_point / dividend

                visit_point = math.sqrt(2.0 * math.log(self.n_visits + 1.0) / n_visit_value)

                rand_point = TreeNode.r.random() * TreeNode.epsilon
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
    

    def rollout(self, temp_node, arr_pred_clearing_price: list[float], ob: Observer, needed_mwh: float, ini_needed_energy: float, actions: list[Action], mcts: MCTS) -> tuple[float, float]:
        tn = TreeNode(temp_node)  # Create a copy of the node

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
                if tn.action_type == Action.ACTION_TYPE.BUY:
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
                    bprice = abs(ob.market_manager.get_balancing_price(min_mwh))  # Replace with actual method call
                    bpriceerr = ob.market_manager.get_mvn_avg_bal_err()  # Replace with actual method name
                    bprice += bpriceerr / 2.0 if bpriceerr > 0.0 else -bpriceerr * 2.0
                    min_mwh /= number_of_bids
                    for i in range(1, int(number_of_bids) + 1):
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
    

    def simulation(self, arr_pred_clearing_price: list[float], ob: Observer, needed_mwh: float, ini_needed_energy: float) -> tuple[float, float]:
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
            if self.action_type == Action.ACTION_TYPE.BUY: 
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
                bprice = abs(ob.market_manager.get_balancing_price(min_mwh)) 
                bpriceerr = ob.market_manager.get_mvn_avg_bal_err() 
                bprice += bpriceerr / 2.0 if bpriceerr > 0.0 else -bpriceerr * 2.0
                min_mwh /= number_of_bids
                for i in range(1, int(number_of_bids) + 1):
                    if bprice <= clearing_price:
                        cost_value -= min_mwh * clearing_price
                        total_bid_volume -= min_mwh
                    bprice += unit_price_increment

            if self.hour_ahead_auction == ob.hour_ahead:
                self.current_node_cost_last = -1.0 * total_bid_volume * clearing_price

        return total_bid_volume, cost_value


    def update_stats(self, sim_cost: float, balancing_sim_cost: float) -> None:
        self.tot_value = ((self.tot_value * self.n_visits + sim_cost) / (self.n_visits + 1.0))
        self.current_node_cost_avg = ((self.current_node_cost_avg * self.n_visits + self.current_node_cost_last) / (self.n_visits + 1.0))
        self.n_visits += 1.0


    def arity(self) -> int:
        return 0 if self.children is None else len(self.children)  # Pythonic way to check for empty list


    def get_children(self, hour_ahead: int, count: int = 0):
        if count == hour_ahead:
            return self.children
        return self.get_children(hour_ahead, count + 1)