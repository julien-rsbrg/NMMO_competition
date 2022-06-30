from CompetitiveEnv.env import CompetitionMultiAgentEnv
from utils import *

import random

#My parameters
n_teams = 2
n_soldiers = 2
do_render = True
only_one_episode = True

#Define env
env = CompetitionMultiAgentEnv(n_teams, n_soldiers)


#Reset and get first observations
observations = env.reset()
#Multi-agents (multi-teams) environment loop
timestep = 0
while True:
    
    #From observations, decide actions
    actions = {}    
    for agent_id, obs in observations.items():
        actions[agent_id] = random.randint(0, n_teams * n_soldiers - 1)   # 0 is for killing itself, 1 for shooting next player, 2 for shooting next next player, ...

    #Everyone do actions, get observations
    (   
        observations, 
        rewards, 
        dones, 
        infos,
    ) = env.step(actions)
    if dones['__all__']:
        if do_render: env.render()
        if only_one_episode : break
        observations = env.reset()
        
    #Render
    timestep += 1
    print("Timestep: ", timestep)
    if do_render: env.render()