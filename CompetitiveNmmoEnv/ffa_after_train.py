import sys
from ijcai2022nmmo import CompetitionConfig, TeamBasedEnv
from ijcai2022nmmo.evaluation.team import Team
from ijcai2022nmmo.scripted import RandomTeam, ForageTeam, CombatTeam

from ray.rllib.agents.ppo import ppo
from CompetitiveNmmoEnv.MA_train import MA_ENV_CLASS, MA_ENV_NAME, MA_ENV_CONFIG, MA_ENV_CONFIG_TEST, config
from CompetitiveNmmoEnv.translators import *

# My parameters
n_teams = 16
n_soldiers = 8
do_render = True


# Create the env from a config
class Config(CompetitionConfig):
    RENDER = do_render


env = TeamBasedEnv(Config())
# Print config eventually



# Define teams
class NotMovingTeam(Team):
    def act(self, observations):
        actions = {}
        for player_idx, obs in observations.items():
            actions[player_idx] = {}
        return actions

class RestoredFromCheckpointTeam(Team):
    #Class parameters:
    checkpoint_path = "tmp/ppo/NMMO_MA/checkpoint_000001/checkpoint-1"
    MA_ENV_NAME_ = MA_ENV_NAME  #the name of the MA env registered that corresponds to the TeamBased 2 MAenv wrapper.
    config_ = config            #same config as the training 
    policy_mapping_fn = config["multiagent"]["policy_mapping_fn"]       # the function that maps the soldier id to the policy id.
    obs2obsUsefull = ObservationToZero()
    actionUsefull2action = ActionUsefullToNoAction()
    
    #Do not modify below
    agent = ppo.PPOTrainer(config_, MA_ENV_NAME_)
    agent.restore(checkpoint_path)
    
    def act(self, observations):
        actions = {}
        for player_idx, observation_soldier in observations.items():
            observation_usefull = self.obs2obsUsefull.process(observation_soldier)
            actionUsefull = self.agent.compute_action(observation_usefull, policy_id = self.policy_mapping_fn(player_idx))
            actions[player_idx] = self.actionUsefull2action.traduce(actionUsefull)
            print("Observation soldier:", observation_soldier.keys())
            print("Observation usefull:", observation_usefull)
            print("Action usefull:", actionUsefull)
            print("Action soldier:", actions[player_idx])
            print()
        return actions

team_classes = [RestoredFromCheckpointTeam] * 15 + [RestoredFromCheckpointTeam]

teams = [Team(team_id="team"+str(nbr_team), env_config=env.config) for nbr_team, Team in enumerate(team_classes)]


# Reset and get first observations
observations_by_team = env.reset()
# Multi-agents (multi-teams) environment loop
timestep = 0  # there is already Tick, so it is not very useful



while True:

    # From observations, decide actions
    actions_by_team = {}
    surviving_teams = []
    for nbr_team, team_observation in observations_by_team.items():
        surviving_teams.append(nbr_team)
        actions_by_team[nbr_team] = teams[nbr_team].act(team_observation)

    print("Surviving team ids:", surviving_teams)
    # Everyone do actions, get observations
    (
        observations_by_team,
        rewards_by_team,
        dones_by_team,
        infos_by_team,
    ) = env.step(actions_by_team)

    # Render
    timestep += 1
    print("Timestep: ", timestep)
    # print(rewards_by_team[0])
    if do_render:
        env._env.render()
