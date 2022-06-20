from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env

from PD.utils import *

from typing import *

#A multi agent env for the Prisoner's Dilemma problem.
class PD_MultiAgentEnv(MultiAgentEnv):
        
    def __init__(self, n_players):
        self.n_players = n_players
        self._agent_ids = list(range(n_players))
        self.last_actions = None    #for render()
        self.observation_space = None #bug

    def reset(self) -> dict[int, Obs]:
        """Reset the environment.

        Returns:
            dict[int, Obs]: key correspond to VirtualAgent ID. Value is the observation of the agent.
            A VirtualAgent correspond to a policy that will be used to perform a certain task in the environment. Several VA forms a PhysicalAgent, or an Agent.
        """
        return {"prisoner"+str(i): 0 for i in range(self.n_players)}
        
    def step(self, actions: dict[str, Action]) -> Tuple[dict[int, Obs], dict[int, float], dict[int, bool], dict[int, dict]]:
        
        #1 - get actions from RLlib policies and execute in your environment
        self.last_actions = actions

        #2 - update the env-state of the environment
        C = 100 / (1.5 ** sum(actions.values())     )                               #betraying divide C the common reward by 1.5
        
        #3 give relevant info to the agent (obs and reward)
        observations =  {"prisoner"+str(i): 0 for i in range(self.n_players)}
        rewards =       {"prisoner"+str(i): C * (1+actions["prisoner"+str(i)]) for i in range(self.n_players)}            #betraying double C your reward
        dones =         {"prisoner"+str(i): True for i in range(self.n_players)}
        dones['__all__'] = True
        infos =         {"prisoner"+str(i): {} for i in range(self.n_players)}
        return observations, rewards, dones, infos
    
    def close(self):
        pass
    def render(self, mode='human'):
        print("Actions chosen:", self.last_actions)



#This will register the MA PD env as a gym env, and will be available in the gym registry. 
def env_creator(env_config):
    n_players = env_config["n_players"]
    return PD_MultiAgentEnv(n_players=n_players)
register_env("multiagent_PrisonnerDilemma", env_creator)