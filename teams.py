from nmmo.io import action
from ijcai2022nmmo import Team

class NotMovingTeam(Team):
    def __init__(self, team_id: str, config = None, **kwargs):
        super().__init__(team_id, config)

    def act(self, observations):
        actions = {}
        for player_idx, obs in observations.items():
            actions[player_idx] = {}
        return actions

#WIP
class GoNorthTeam(Team):
    def __init__(self, team_id: str, config = None, **kwargs):
        super().__init__(team_id, config)

    def act(self, observations):
        actions = {}
        for player_idx, obs in observations.items():
            actions[player_idx] = {}
        return actions

