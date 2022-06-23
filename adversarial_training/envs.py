"""
This gym environment is a variation of the Prisoner's Dilemma for N player.
Training adversarial agents on this environment would result to systematic betrayal,
while training cooperative agents on this environment would result to systematic cooperation.

Each agent receive a proportion of C the total quantity of food. 
Betraying reduce C but increase the proportion of food for the agent.
"""

import gym
from ray.tune.registry import register_env


import math
import random
from typing import List, Dict, Tuple, Union

from adversarial_training.agents import Agent, RandomAgent
from adversarial_training.utils import Obs, Action
from adversarial_training.ma_envs import PD_MultiAgentEnv

    
#From a MultiAgent env, create and individual env.
class IndividualEnvWrapper(gym.Env):
    
    def __init__(self, MA_env, agents : List[Union[Agent, None]], agent_idx : int):
        super().__init__()
        self.MA_env = MA_env
        self.agents = agents
        self.agent_idx = agent_idx
        self.observations : dict[int, Obs] = None           #observations of all agents
        self.observation_space = gym.spaces.Discrete(1)
        self.action_space = gym.spaces.Discrete(2)
                
    def reset(self, seed : int = 0) -> Obs:
        self.observations = self.MA_env.reset()
        return self.observations[self.agent_idx]
    
    def step(self, action : Action) -> Tuple[Obs, int, bool, dict]:
        actions = {self.agent_idx: action}
        for i in range(len(self.agents)):
            if i != self.agent_idx:
                actions[i] = self.agents[i].act(self.observations[i])
        self.observations, rewards, dones, infos = self.MA_env.step(actions)
        return self.observations[self.agent_idx], rewards[self.agent_idx], dones[self.agent_idx], infos[self.agent_idx]

    def render(self, mode='human'):
        pass
    def close(self):
        pass




#This will register the PD env as a gym env, and will be available in the gym registry. 
def env_creator(env_config):
    agents = env_config["agents"]
    agent_idx = env_config["agent_idx"]
    n_players = len(agents)
    return IndividualEnvWrapper(MA_env=PD_MultiAgentEnv(n_players), agents = agents, agent_idx=agent_idx)  # return an env instance
register_env("PD", env_creator)









    
if __name__ == '__main__':
                                                        # Test : run the betrayal policy (action = 1) for 10 turns.
    MA_env = PD_MultiAgentEnv(n_players=4)
    env = IndividualEnvWrapper(MA_env, [RandomAgent(gym.spaces.Discrete(2)) for _ in range(4)], agent_idx=0)
    observation = env.reset(seed=42)

    for _ in range(10):
        env.render()
        action = 1
        observation, reward, done, info = env.step(action)
        print("Timestep: ", _)
        print("Observation: ", observation)
        print("Reward: ", reward)
        if done:
            observation = env.reset()
    env.close()
    print("Done")