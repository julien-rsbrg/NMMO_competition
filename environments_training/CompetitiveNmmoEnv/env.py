from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env
import gym
from gym import spaces
from gym.spaces import Dict, Discrete, MultiDiscrete, Tuple

import nmmo
from ijcai2022nmmo import CompetitionConfig, TeamBasedEnv

from utils import *
from CompetitiveNmmoEnv.translators import *

from typing import Dict, Union
import typing

soldier_observation_space = Discrete(1)
soldier_action_space = Discrete(3)


class CompetitionNmmoMultiAgentEnv(MultiAgentEnv):

    def __init__(self, Config_class: CompetitionConfig):
        self.n_teams = 16
        self.n_soldiers = 8
        self.team_based_env: TeamBasedEnv = TeamBasedEnv(config=Config_class())
        self._agent_ids = [str(i) + '_' + str(j)
                           for i in range(self.n_teams) for j in range(self.n_soldiers)]
        self.translators = {num_team: {num_soldier: {"obs2obsUsefull": ObservationToZero(),
                                                     "actionUsefull2action": ActionUsefullToNoAction()}
                                       for num_soldier in range(self.n_soldiers)} for num_team in range(self.n_teams)}

        # Define spaces
        self.observation_space = soldier_observation_space
        self.action_space = soldier_action_space

    def reset(self, seed=0) -> dict[AgentID, Obs]:
        observations_by_team = self.team_based_env.reset()
        observations_usefull = {}  
        for num_team, observations_by_soldiers in observations_by_team.items():
            for num_soldier, observation in observations_by_soldiers.items():
                agent_id = str(num_team) + '_' + str(num_soldier)
                observation_usefull = self.translators[num_team][num_soldier]["obs2obsUsefull"].process(
                    observation)
                observations_usefull[agent_id] = observation_usefull
        return observations_usefull

    def step(self, actions: dict[AgentID, Action]) -> typing.Tuple[dict[AgentID, Obs], dict[AgentID, float], dict[AgentID, bool], dict[AgentID, dict]]:

        actions_by_team = {}
        for agent_id, actionUsefull in actions.items():
            num_team, num_soldier = self.id_to_numbers(agent_id)
            action = self.translators[num_team][num_soldier]["actionUsefull2action"].traduce(
                actionUsefull)
            if num_team not in actions_by_team:
                actions_by_team[num_team] = {}
            actions_by_team[num_team][num_soldier] = action

        (
            observations_by_team,
            rewards_by_team,
            dones_by_team,
            infos_by_team,
        ) = self.team_based_env.step(actions_by_team)

        observations_usefull = {}
        rewards = {}  # !!
        dones = {}
        infos = {}
        for num_team, observations_by_soldiers in observations_by_team.items():
            for num_soldier, observation in observations_by_soldiers.items():
                agent_id = str(num_team) + '_' + str(num_soldier)
                observation_usefull = self.translators[num_team][num_soldier]["obs2obsUsefull"].process(
                    observation)
                observations_usefull[agent_id] = observation_usefull
                rewards[agent_id] = rewards_by_team[num_team][num_soldier]  # !!
                dones[agent_id] = dones_by_team[num_team][num_soldier]
                infos[agent_id] = infos_by_team[num_team][num_soldier]

        # all done when litterally all done
        dones["__all__"] = all(dones.values())
        return observations_usefull, rewards, dones, infos

    def render(self, mode: str = "human") -> None:
        self.team_based_env._env.render()

    def close(self):
        pass

    # Non interface methods below
    def id_to_numbers(self, agent_id: AgentID):
        num_team, num_soldier = agent_id.split("_")
        return int(num_team), int(num_soldier)


# Register env
def env_creator(env_config):
    Config_class = env_config['Config_class']
    return CompetitionNmmoMultiAgentEnv(Config_class=Config_class)


register_env("CompetitionNmmoMultiAgentEnv", env_creator)
