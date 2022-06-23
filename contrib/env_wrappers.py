"""
This gym environment is a variation of the Prisoner's Dilemma for N player.
Training adversarial agents on this environment would result to systematic betrayal,
while training cooperative agents on this environment would result to systematic cooperation.

Each agent receive a proportion of C the total quantity of food. 
Betraying reduce C but increase the proportion of food for the agent.
"""

import gym
from ray.tune.registry import register_env
from ray.rllib.env.multi_agent_env import MultiAgentEnv

import math
import random
from typing import List, Dict, Tuple, Union

from contrib.actors import Actor, RandomActor
from utils import Obs, Action

    
#From a MultiAgent env, create and individual env.
class IndividualEnvWrapper(gym.Env):
    """
    Turn a MultiAgentEnv into a gym.Env.
    """
    
    def __init__(self, 
                        MA_env : MultiAgentEnv, 
                        actors : Dict[str, Actor], 
                        agent_id : str):
        """
        MA_env : the MultiAgentEnv to wrap
        actors : the Actor object to use for each agent, as a dictionnary mapping agent_id with Actor object. An actor object must have an act method. 
        agent_id : the agent_id in the MA env that we want to choose as our agent for this gym env.
        """
        super().__init__()
        self.MA_env = MA_env
        self.actors = actors
        self.agent_id = agent_id
        self.observations : dict[str, Obs] = None           #observations of all agents
                
    def reset(self, seed : int = 0) -> Obs:
        self.observations = self.MA_env.reset()
        while self.agent_id not in self.observations:
            actions = {}
            for i, obs in self.observations.items():
                actions[i] = self.actors[i].act(obs)
            self.observations, rewards, dones, infos = self.MA_env.step(actions)
        return self.observations[self.agent_id]
    
    def step(self, action : Action) -> Tuple[Obs, float, bool, dict]:
        #Agent take action as well as other agents
        agent_obs = self.observations[self.agent_id]
        actions = {self.agent_id: action}
        for i, obs in self.observations.items():
            if i != self.agent_id:
                actions[i] = self.actors[i].act(obs)
        self.observations, rewards, dones, infos = self.MA_env.step(actions)
        #If our agent does not receive observation, it means it is not its turn, so we keep playing.
        while self.agent_id not in self.observations:
            if dones['__all__']:
                return agent_obs, 0, True, {}   #for the final step if agent has no access to observation, the next_obs returned is the obs (because it should not be taken into account anyway), could also return None but may cause bugs I guess
            actions = {}
            for i, obs in self.observations.items():
                actions[i] = self.actors[i].act(obs)
            self.observations, rewards, dones, infos = self.MA_env.step(actions)
        #When our agent receives observation, it is its turn, so we can take action.
        # print("Printing elements:", self.observations, rewards, dones, infos, sep = '\n')
        info = infos[self.agent_id] if self.agent_id in infos else {}
        return self.observations[self.agent_id], rewards[self.agent_id], dones[self.agent_id], info

    def render(self, mode='human'):
        self.MA_env.render()
    def close(self):
        pass

