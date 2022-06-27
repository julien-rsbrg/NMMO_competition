from nmmo import Task, Env
from ijcai2022nmmo import CompetitionConfig, TeamBasedEnv

# ==================== Modify the reward by tasks ==================== #


def player_kills(realm, player):
    return player.history.playerKills


def equipment(realm, player):
    return player.loadout.defense


def exploration(realm, player):
    return player.history.exploration


def foraging(realm, player):
    return (player.skills.fishing.level + player.skills.hunting.level) / 2.0


PlayerKills = [
    Task(player_kills, 1, 10),
    Task(player_kills, 2, 10),
    Task(player_kills, 3, 10),
    Task(player_kills, 4, 10),
    Task(player_kills, 5, 10),
    Task(player_kills, 6, 10),
]

Equipment = [
    Task(equipment, 1, 2),
    
]

# Exploration = [
#     Task(exploration, 32, Tier.EASY),
#     Task(exploration, 64, Tier.NORMAL),
#     Task(exploration, 127, Tier.HARD)
# ]

# Foraging = [
#     Task(foraging, 20, Tier.EASY),
#     Task(foraging, 35, Tier.NORMAL),
#     Task(foraging, 50, Tier.HARD)
# ]

rewardingTasks = []

# =============================================================== #






#Modify reward by wrapping the env
class RewardEnv(Env):
    
    def reward(self, player):
        reward, info = super().reward(player)
        
        # =============== Modify reward here ===============#
        
        # ================================================== #

        return reward, info
    

# DO NOT MODIFY BELOW 
class RewardTeamBasedEnv(TeamBasedEnv):
    def __init__(self, config : CompetitionConfig) -> None:
        config.TASKS = rewardingTasks
        self._env = RewardEnv(config)