from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env
import gym
from utils import *

from typing import *



class CompetitionMultiAgentEnv(MultiAgentEnv):
    hp_max = 5
                                
    def __init__(self, n_teams, n_soldiers):
        ### MAENV attributes
            # agent of soldier 5 of team 1 is 1_5
        self._agents_ids = [str(i) + '_' + str(j) for i in range(n_teams) for j in range(n_soldiers)]
        # self.observation_space = gym.spaces.Box(low=0, high=10, shape=(n_teams * n_soldiers,), dtype=np.float32) #for bug
        # self.action_space = gym.spaces.Discrete(n_teams * n_soldiers) # for bug
        self.observation_space = None
        
        ### Particular to this env attributes
        self.n_teams = n_teams
        self.n_soldiers = n_soldiers
        self.last_actions = {}
        
    def reset(self, seed = 0) -> dict[int, Obs]:
        """
        Reset the env and return the initial observations for all ready agents (ie all of them for this initialization).
        """
        self.healths = {agent_id: hp for agent_id in self._agents_ids}
        return {agent_id: self.health_to_obs(num_teams=self.id_to_numbers(agent_id)[0], healths = self.healths) for agent_id in self._agents_ids}
    
    def step(self, actions: dict[str, Action]) -> Tuple[dict[int, Obs], dict[int, float], dict[int, bool], dict[int, dict]]:
        """
        Apply actions and return observations, rewards, dones for all ready agents (ie all living agents) as well as infos for all agents.
        """
        self.last_actions = actions
        rewards = {}
        
        #Each soldier alive shoot a target (remove 1 hp) and get a reward for shooting an enemy
        for num_team in range(self.n_teams):
            for num_soldier in range(self.n_soldiers):
                agent_id = str(num_team) + '_' + str(num_soldier)
                if self.healths[agent_id] > 0:
                    num_team, num_soldier = self.id_to_numbers(agent_id)
                    target_action_perspective = actions[agent_id]
                    target_action = (target_action_perspective + num_team * self.n_soldiers) % (self.n_teams * self.n_soldiers)
                    target_id = str(target_action // self.n_soldiers) + '_' + str(target_action % self.n_soldiers)
                    #Remove 1 hp to the target and get a reward if the target was a living enemy. Recevie a penalty if shooting an ally.
                    self.healths[target_id] -= 1
                    if target_action_perspective <= self.n_soldiers - 1:
                        rewards[agent_id] = -2 
                    elif self.healths[target_id] >= 0:
                        rewards[agent_id] = 1
                    else:
                        rewards[agent_id] = 0
                    self.healths[target_id] = max(0, self.healths[target_id])
        
        #Check dones
        dones = {}
        num_teams_alive = 0
        for num_team in range(self.n_teams):
            team_is_alive = False
            for num_soldier in range(self.n_soldiers):
                agent_id = str(num_team) + '_' + str(num_soldier)
                if self.healths[agent_id] <= 0:
                    dones[agent_id] = True
                else:
                    dones[agent_id] = False
                    team_is_alive = True
            if team_is_alive:
                num_teams_alive += 1
        if num_teams_alive <= 1:
            dones = {key : True for key in self._agents_ids}
            dones['__all__'] = True
        else:
            dones['__all__'] = False
        
        #observations
        observations = {}
        rewards_for_ready_agents = {} 
        dones_for_ready_agents = {'__all__': dones['__all__']}
        for agent_id, is_done in dones.items():
            if not is_done and agent_id != '__all__':
                observations[agent_id] = self.health_to_obs(num_teams=self.id_to_numbers(agent_id)[0], healths = self.healths)
                rewards_for_ready_agents[agent_id] = rewards[agent_id]
                dones_for_ready_agents[agent_id] = dones[agent_id]
              
        #infos
        infos = {agent_id : {} for agent_id in self._agents_ids}
        
        #returns
        return observations, rewards_for_ready_agents, dones_for_ready_agents, infos
    
    def render(self):
        print("RENDER:")
        print("LAST ACTIONS: ", self.last_actions)
        print("HEALTHS: ", self.healths, "\n")
        
    def close(self):
        pass
    
    
    
    #Non MAEnv interface methods belows:
    
    def health_to_obs(self, num_teams, healths):
        #From a healths dict, return an obs dict from the num_teams perspective
        list_healths = [healths[str(i) + '_' + str(j)] for i in range(self.n_teams) for j in range(self.n_soldiers)] 
        list_healths_perspective = self.list_to_perspective_list(list_healths, num_teams * self.n_soldiers)
        return np.array(list_healths_perspective)
        
    def list_to_perspective_list(self, l, n = 0):
        return l[n:] + l[:n]

    def id_to_numbers(self, agent_id):
        num_team, num_soldier = agent_id.split("_")
        return int(num_team), int(num_soldier)




### Register env
def env_creator(env_config):
    n_teams = env_config['n_teams']
    n_soldiers = env_config['n_soldiers']
    return CompetitionMultiAgentEnv(n_teams, n_soldiers)
register_env("CompetitionMultiAgentEnv", env_creator)




### Register individualized env (against random)
from contrib.env_wrappers import IndividualEnvWrapper
from contrib.actors import RandomActor
def env_creator(env_config):
    n_teams = env_config['n_teams']
    n_soldiers = env_config['n_soldiers']
    soldier_action_space = gym.spaces.Discrete(n_soldiers * n_teams)
    MA_env = CompetitionMultiAgentEnv(n_teams, n_soldiers)
    return IndividualEnvWrapper(MA_env = MA_env, 
                                actors={agent_id : RandomActor(action_space=soldier_action_space) for agent_id in MA_env._agents_ids},
                                agent_id="0_0",
                                )
register_env("SingleAgentAgainstRandomCompetitionMultiAgentEnv", env_creator)
