from ijcai2022nmmo import CompetitionConfig, scripted, submission, RollOut
from teams import NotMovingTeam
import nmmo

n_teams= 16
class TeamBattleConfig(CompetitionConfig):
    NPOP = n_teams
    NENT = 8 * n_teams

    @property
    def SPAWN(self):
        return self.SPAWN_CONCURRENT

    AGENT_LOADER = nmmo.config.TeamLoader

    AGENTS = NPOP * [nmmo.Agent]

    NMAPS = 1
    RENDER = True
    PATH_MAPS = "maps"
    
config = TeamBattleConfig()

# my_team = submission.get_team_from_submission(
#     submission_path="my-submission/",
#     team_id="MyTeam",
#     env_config=config,
# )

# Or initialize the team directly
my_team = NotMovingTeam("Myteam", config = config)

teams = []
teams.append(my_team)
teams.extend([scripted.CombatTeam(f"Combat-{i}", config) for i in range(3)])
teams.extend([scripted.ForageTeam(f"Forage-{i}", config) for i in range(5)])
teams.extend([scripted.RandomTeam(f"Random-{i}", config) for i in range(7)])
teams = teams[:n_teams]



if __name__ == "__main__":
    ro = RollOut(config, teams, parallel=True, show_progress=True)
    ro.run(n_episode=1)