from broker.spot.action_bu import Action
from broker.spot.mcts import MCTS

class Bidding_Strategy(object):

    server_min_mwh = 0.01
    min_limit_mwh = 2.0
    start_timeslot = 360
    wholesale_current_timeslot = 360
    
    first_time_flags = True

    def __init__(self):
        self.mcts = MCTS()

    def version(self):
        return "SPOT's Bidding Strategy"
    
    def action(self, currentTimeslot,):
         self.wholesaleCurrentTimeslot = currentTimeslot

    def submit_order(self, needed_kwh, timeslot):
        try:
            balancing_price = 0
            current = 360   # check
            
            remaining_tries = timeslot - current
            
            days = (current - self.start_timeslot) / 24
            hour = (current - self.start_timeslot) % 24
            
            needed_mwh = needed_kwh / 1000.0
            
            if self.first_time_flags == True:
                self.mcts.setup()
                self.first_time_flags = False
            
            if remaining_tries >= 24: 
                print(f"remainingTries >= 24 i.e. {remaining_tries} setting to 23")
                remainingTries = 23
            elif remainingTries < 0:                
                print(f"remainingTries >= 24 i.e. {remaining_tries} setting to 0")
                remainingTries = 0

            self.observer.set_time(days, hour, remaining_tries, current)
      
            posn = self.broker.getBroker().findMarketPositionByTimeslot(timeslot)
            
            if posn != None:
                neededMWh -= posn.getOverallBalance()
            
            self.observer.needed_enery_per_broker = needed_mwh
            self.observer.initial_needed_enery_mcts_broker = needed_mwh
            
            if abs(needed_mwh) <= self.server_min_mwh:
                return
            
            bday = days - 1
      
            balancing_price_obj = self.balancing_price_history.get(Integer.valueOf(bday))
            
            if balancing_price_obj == None:
                balancing_price = 90.0
            else:
                blc_price = balancing_price_obj.get_balancing_price_from_history()
                if blc_price == None:
                    balancing_price = 90.0
                else:                
                    balancing_price = abs((blc_price.meanprice + blc_price.maxprice) / 2)
                
                    if balancing_price <= 0.0:
                        balancing_price = 90.0

            self.observer.balacing_price = abs(balancing_price)
            market_limit_price = self.market_clearing_price_prediction.get(Integer.valueOf(days))
            moving_avg_err = self.observer.market_manager.get_mvn_avg_err()
            
            for haid in range(24):
                predicted_clearing_price_per_auction = 30.0
                
                if market_limit_price != None:                
                    if (market_limit_price.predicted_clearing_price_per_auction[hour][haid] != 0.0):
                        predicted_clearing_price_per_auction = market_limit_price.predicted_clearing_price_per_auction[hour][haid]
                self.observer.arr_clearing_prices[haid] = predicted_clearing_price_per_auction - moving_avg_err

            best_move = self.mcts.get_best_mcts_move(self.observer)

            number_of_bids = 10
            unit_price_increment = 1
            limit_price = abs(best_move.minmcts_clearing_price)
            price_range = abs(best_move.maxmcts_clearing_price - best_move.minmcts_clearing_price)
            
            min_mwh = neededMWh / number_of_bids
            if abs(min_mwh) <= self.server_min_mwh:
                min_mwh = neededMWh
                number_of_bids = 1 
            
            unit_price_increment = price_range / number_of_bids
            
            if best_move.nobid == False:               
                if best_move.actionType == Action.ACTION_TYPE.BUY:
                    if needed_mwh > 0.0:
                        surplus = abs(needed_mwh) * (best_move.vol_percentage - 1.0)
                        totalE = surplus + abs(needed_mwh)
                        min_mwh = totalE / number_of_bids
                        
                        if limit_price > 0:
                            limit_price *= -1.0
                            unit_price_increment *= -1.0
                        
                        b = self.broker.getBroker()
                        for i in range (number_of_bids+1):
                            order = new Order(b, timeslot, min_mwh, limit_price)
                            self.broker.send_message(order)
                            limit_price += unit_price_increment
                else:
                    surplus = -needed_mwh
                    min_mwh = abs(needed_mwh) * (best_move.vol_percentage - 1)
                    if min_mwh < surplus:
                        min_mwh = surplus
        
                    bprice = abs(self.observer.marketManager.getBalancingPrice(minMWh))
                    bpriceerr = self.observer.marketManager.getMvnAvgBALErr()

                    if bpriceerr > 0.0:
                        bprice += bpriceerr / 2.0
                    else:                       
                        bprice -= bpriceerr * 2.0
                    
                    min_mwh /= number_of_bids
                    
                    if min_mwh > 0.0:
                        min_mwh *= -1.0
                    
                    if bprice < 0.0:
                        bprice *= -1.0
                    
                    if unit_price_increment < 0.0:
                        unit_price_increment *= -1.0
            
                    b = self.broker.getBroker()
                    for i in range (number_of_bids+1):
                        Order order = new Order(b, timeslot, minMWh, Double.valueOf(bprice))
                        self.broker.sendMessage(order)
                        bprice += unit_price_increment
                        
        except Exception as E:
            print(E)