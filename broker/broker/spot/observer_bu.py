import io

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
        self.balacing_price = 90.0
        self.needed_energy_per_broker = 0.0
        self.initial_needed_energy_mcts_broker = 0.0
        self.STDDEV2016 = 13.38
        self.MEANSTDDEV2016 = -0.24
        self.STDDEV2017 = 17.68961
        self.MEAN2017 = 39.50736
        self.STDDEV = self.STDDEV2016

        try:
            with io.open("activeWholesalePredictorModel.txt", "r") as file:
                predictor_version = file.readline().strip()
                self.STDDEV = float(file.readline().strip())
        except Exception as e:
            print("Unable to read stddev value:", e)
            self.STDDEV = self.STDDEV2016

    def set_time(self, day, hour, hour_ahead, current_time_slot):
        self.day = day
        self.hour = hour
        self.hour_ahead = hour_ahead
        self.current_time_slot = current_time_slot

    def print(self):
        print("day", self.day, "hour", self.hour, "hourAhead", self.hour_ahead, "currentTimeSlot", self.current_time_slot)
