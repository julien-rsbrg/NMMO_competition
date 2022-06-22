from ijcai2022nmmo import CompetitionConfig, TeamBasedEnv
from ijcai2022nmmo.evaluation.team import Team
from ijcai2022nmmo.scripted import RandomTeam, ForageTeam, CombatTeam
from utils import *

#My parameters
n_teams = 16
n_soldiers = 8
do_render = False


#Create the env from a config
class Config(CompetitionConfig):
    RENDER = do_render
env = TeamBasedEnv(Config())
#Print config eventually
def printConfig(config):
    for attr in dir(config):
        if not attr.startswith('__'):
            print('{}: {}'.format(attr,getattr(config,attr)))            
printConfig(env.config)


#Define teams
class NotMovingTeam(Team):
    def act(self, observations):
        actions = {}
        for player_idx, obs in observations.items():
            actions[player_idx] = {}
        return actions

team_classes = [CombatTeam for nbr_team in range(n_teams-1)] + [NotMovingTeam] 

teams = [Team(team_id="team"+str(nbr_team), env_config=env.config) for nbr_team, Team in enumerate(team_classes)]



#Reset and get first observations
observations_by_team = env.reset()
#Multi-agents (multi-teams) environment loop
timestep = 0
while True:
    
    #From observations, decide actions
    actions_by_team = {}    
    for nbr_team, team_observation in observations_by_team.items():
        actions_by_team[nbr_team] = teams[nbr_team].act(team_observation)

    #Everyone do actions, get observations
    (   
        observations_by_team, 
        rewards_by_team, 
        dones_by_team, 
        infos_by_team,
    ) = env.step(actions_by_team)
    
    #Render
    print(obs_dict_to_string(observations_by_team[5]))
    raise
    timestep += 1
    print("Timestep: ", timestep)
    # print(rewards_by_team[0])
    if do_render: env._env.render()