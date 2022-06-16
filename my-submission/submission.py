import random
from nmmo.io import action

from ijcai2022nmmo import Team, scripted


class RandomTeam(Team):
    def act(self, observations):
        actions = {}
        for player_idx, obs in observations.items():
            actions[player_idx] = {
                action.Move: {
                    action.Direction:
                    random.randint(0,
                                   len(action.Direction.edges) - 1)
                }
            }
        return actions

CombatTeam = scripted.CombatTeam

class Submission:
    team_klass = CombatTeam
    init_params = {}
