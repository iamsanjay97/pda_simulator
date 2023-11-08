import gym
import math

from gym import spaces
from collections import OrderedDict

## A Naive Congestion Pricing Genco model

class CPGenCoEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4, "name": "CPGenCo-v0"}

    def __init__(self, render_mode=None):
        self.seed_value = 0 # seed value will be overidden
        self.id = None
        self.total_quantity = 1500 
        self.cleared_quantity = 0
        self.knee_demand = 900 # Congesiton threshold
        self.knee_slope = 5
        self.min_ask_quant = 15
        self.a = 0.00004132
        self.b = 0.00182
        self.c = 14
        self.p = 6

        self.observation_space = spaces.Box(0, 1, shape=(2,), dtype=float)
        self.action_space = spaces.Discrete(4)
        self.render_mode = render_mode

    def set_id(self, id):
        self.id = id
        
    def scale_coeff(self):
        y = self.a * self.knee_demand *self.knee_demand + self.b * self.knee_demand + self.c
        self.a *= self.knee_slope
        self.b *= self.knee_slope
        self.c = y - self.a * self.knee_demand * self.knee_demand - self.b * self.knee_demand        
        self.p *= 2
 
    def quad_function(self, cum_quantity):
        a,b,c,p = self.a,self.b,self.c,self.p
        if cum_quantity >= self.knee_demand: 
            self.scale_coeff()
                    
        q = cum_quantity
        price = self.a*(q**2)+self.b*q+self.c
        quantity = max((-self.b/(2.0*self.a) + math.sqrt(self.b**2+4.0*self.a*(self.a*(q**2)+self.b*q+self.p))/(2.0*self.a) - q) , self.min_ask_quant)
        self.a,self.b,self.c,self.p = a,b,c,p
        return price, quantity
    
    def asks(self):
        asks = list()
        cum_quantity = self.cleared_quantity
        while cum_quantity < self.total_quantity:
            price, quantity = self.quad_function(cum_quantity)
            cum_quantity += quantity
            asks.append([self.id, price, -quantity])

        return asks
    
    def set_cleared_quantity(self, cleared_quantity):
        self.cleared_quantity += cleared_quantity

    def reset(self):
        pass

    def step(self):
        pass

    def render(self):
        pass

    def close(self):
        pass

    