"""
Implementation of different memory functions for the DQN learner.
Two different memories are available:
- The standard implementation of experience replay
- The implementation of prioritized experience replay
 
@author: Felix Strnad
"""

import numpy as np

import random
from DeepReinforcementLearning.SumTree import SumTree
from collections import deque

""" Basic Memory class, experiences are stored without any logic and drawn randomly"""
class Memory(object):
    samples=deque()    # Deque is faster than list especially for using append and pop. 
    
    def __init__(self, capacity):
        self.capacity = capacity
    
    def add(self, experience):
        self.samples.append(experience)
        if len(self.samples) > self.capacity:
            self.samples.popleft()
    
    """ Returns an random sample of data """
    def sample(self, batch_size):
        buffer_size = min(batch_size, len(self.samples))        
        return None, random.sample(self.samples, buffer_size), None
    
    def isFull(self):
        return len(self.samples) >= self.capacity
    
    
    
"""Prioritized experience replay. Experiences are stored according to their relevance!"""
class PERMemory(object):
    # Parameters for estimating the priority value
    eps = 0.001      # No zero priorities
    # Control parameter, how much prioritization is used. Using alpha= 0, we would get uniformly distributed values
    alp = 0.6
    beta=0.4       
    beta_increment_per_sampling = 1e-4  # annealing the bias
    abs_err_upper = 1 # for stability refer to paper
    
    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _getPriority(self, error):
        priority=(error + self.eps) ** self.alp
        return priority

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample) 

    def sample(self, n):
        batch = []   
        batch_index=[]
        segment = self.tree.total() / n
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)

            batch.append( data )
            batch_index.append(idx)
        
        #print(batch)    
        return batch_index, batch, None # None for weights

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)
        
        
"""
Prioritized experience memory class. 
It can be combined with importance sampling.
For reference see Schaul et al. 2015.
"""    
class PERISMemory(PERMemory):
    # Parameters for estimating the priority value
    beta=0.4        
    beta_increment_per_sampling = 1e-4  # annealing the bias
    abs_err_upper = 1 # for stability refer to paper
    
    def __init__(self, capacity):
        PERMemory.__init__(self, capacity)

    def sample(self, n):
        batch = []   
        batch_index=[]
        ISWeights=[]
        segment = self.tree.total() / n
        self.beta=np.min([1, self.beta + self.beta_increment_per_sampling])
        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total()
        maxiwi = np.power(self.tree.capacity * min_prob, -self.beta) # for later normalizing ISWeights    
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            # Importance sampling  https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/experiments/Solve_LunarLander/DuelingDQNPrioritizedReplay.py
            prob = p / self.tree.total()
            ISWeights.append(self.tree.capacity * prob)
            batch.append( data )
            batch_index.append(idx)
            #print(idx)
        #ISWeights = np.vstack(ISWeights)
        ISWeights = np.power(ISWeights, -self.beta) / maxiwi # normalize
        #print(len(ISWeights), len(batch))
        
        #print(batch)    
        return batch_index, batch, ISWeights        
        
        