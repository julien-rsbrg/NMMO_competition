from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env

from adversarial_training.utils import *
from typing import *

#A multi agent env for the Prisoner's Dilemma problem.
class PD_MultiAgentEnv(MultiAgentEnv):
        
    def __init__(self, n_players):
        self.n_players = n_players
        self.last_actions = None

    def reset(self) -> dict[int, Obs]:
        return {i: 0 for i in range(self.n_players)}
        
    def step(self, actions: dict[int, Action]) -> Tuple[dict[int, Obs], dict[int, int], dict[int, bool], dict[int, dict]]:
        self.last_actions = actions
        
        observations = {i: 0 for i in range(self.n_players)}
        
        C = 100
        C /= 1 + 0.5 * sum(actions.values())  
        rewards = {i: C * (1+actions[i]) for i in range(self.n_players)}
        
        dones = {i: True for i in range(self.n_players)}
        infos = {i: {} for i in range(self.n_players)}
        
        return observations, rewards, dones, infos
    
    def render(self, mode='human'):
        print("Actions chosen:", self.last_actions)
    

#This will register the MA PD env as a gym env, and will be available in the gym registry. 
def env_creator(env_config):
    n_players = env_config["n_players"]
    return PD_MultiAgentEnv(n_players=n_players)
register_env("MA_PD", env_creator)