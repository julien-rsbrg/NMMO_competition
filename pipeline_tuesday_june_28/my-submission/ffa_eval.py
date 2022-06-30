from submission import RestoredFromCheckpointTeam
from CONFIG import ConfigTest
from ijcai2022nmmo.scripted import CombatTeam
from ijcai2022nmmo.evaluation.team import Team
from env import RewardTeamBasedEnv
from ast import Not
print("=========== Script ffa_eval running ===========")
print("Imports...")

print("Import config...")

# My parameters
n_teams = 16
n_soldiers = 8
do_render = True

# Create the env from a config
env = RewardTeamBasedEnv(ConfigTest())


# Define teams
class NotMovingTeam(Team):
    def act(self, observations):
        actions = {}
        for player_idx, obs in observations.items():
            actions[player_idx] = {}
        return actions


class RestoredFromCheckpointTeamPrinting(RestoredFromCheckpointTeam):
    do_print = False


team_classes = 8 * [CombatTeam] + 8 * [RestoredFromCheckpointTeam]
teams = [Team(team_id="team"+str(nbr_team), env_config=env.config)
         for nbr_team, Team in enumerate(team_classes)]

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
    print(rewards_by_team[0])
    if do_render:
        env.render()
