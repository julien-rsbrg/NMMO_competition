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

rewardingTasks = PlayerKills + Equipment

# =============================================================== #






#Modify reward by wrapping the env
class RewardEnv(Env):
    
    def update_stats(self, player):
        player.statistics = {"health" : player.resources.health.val,
                                        "water" : player.resources.water.val,
                                        "food" : player.resources.food.val,
                                        "exploration" : player.history.exploration,
                                    }
    
    
    
    def reward(self, player):
        reward, info = super().reward(player)
        
        # =============== Modify reward here ===============#
        
        #Create statistics for player if has no
        if not hasattr(player, 'statistics'):
            self.update_stats(player)
            return reward, info
            
        #Restore food and water
        dfood = player.resources.food.val - player.statistics["food"]
        if dfood > 0:
            reward += 0.1 * dfood
        dwater = player.resources.water.val - player.statistics["water"]
        if dwater > 0:
            reward += 0.1 * dwater
        #Restore health, lose health, die
        dhealth = player.resources.health.val - player.statistics["health"]
        if player.resources.health.val == 0:
            reward -= 20
        elif dhealth >= 0:
            reward += 0.2 * dhealth
        else:
            reward += 0.5 * dhealth
        #Explore
        dexploration = player.history.exploration - player.statistics["exploration"]
        if dexploration > 0:
            reward += 0.1 * dexploration

        #Update statistics after computing reward
        self.update_stats(player)
        
        # ================================================== #

        return reward, info
    

# DO NOT MODIFY BELOW 
class RewardTeamBasedEnv(TeamBasedEnv):
    def __init__(self, config : CompetitionConfig) -> None:
        config.TASKS = rewardingTasks
        self._env = RewardEnv(config)