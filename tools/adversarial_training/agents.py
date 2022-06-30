"""
Agents are defined as objects that can act.
"""
from adversarial_training.utils import Obs, Action
from abc import ABC, abstractmethod

class Agent(ABC):
    @abstractmethod
    def act(self, observation: Obs) -> Action: pass
        
class RandomAgent(Agent):
    def __init__(self, action_space, sample_func=lambda space: space.sample()):
        self.action_space = action_space
        self.sample_func = sample_func
    def act(self, obs: Obs) -> Action:
        return self.sample_func(self.action_space)
    