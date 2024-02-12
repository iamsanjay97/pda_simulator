import gym
import collections as col
import numpy as np

class PowerTAC_WM(object):

  def __init__(self,proximity=None, balancingPrice=None, quantity=None):

    self.proximity = proximity           
    self.balancingPrice = balancingPrice
    self.quantity = quantity

  def form_network_input(self):

     state = list()
     state.append(self.proximity)
     state.append(self.balancingPrice)
     state.append(self.quantity)

     return state


  def step(self, action):
      pass

  def reset(self):

      self.proximity = 24         
      self.balancingPrice = 0.0
      self.quantity = 0.0
      self.observation = self.make_observaton()
      return self.observation


  def make_observaton(self):

      names = ['proximity', 'balancingPrice', 'quantity']
      Observation = col.namedtuple('Observaion', names)
      return Observation(proximity=self.proximity, balancingPrice=self.balancingPrice, quantity=self.quantity)


  def render(self, mode='human', close=False):

      print('Proximity : ', self.proximity)
      print('Balancing Price : ', self.balancingPrice)
      print('Required Quantity : ', self.quantity)

  def __str__(self):

      return "<" + str(self.proximity) + \
             "," + str(self.balancingPrice) + \
             "," + str(self.quantity) + ">"
