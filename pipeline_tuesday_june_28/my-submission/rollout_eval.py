print("=========== Script rollout_eval running ===========")
from ijcai2022nmmo import CompetitionConfig, scripted, submission, RollOut

config = CompetitionConfig()
from submission import RestoredFromCheckpointTeam

teams = []
teams.extend([scripted.CombatTeam(f"Combat-{i}", config) for i in range(3)])
teams.extend([scripted.ForageTeam(f"Forage-{i}", config) for i in range(5)])
teams.extend([scripted.RandomTeam(f"Random-{i}", config) for i in range(7)])
teams.append(RestoredFromCheckpointTeam("MyTeam", config))

ro = RollOut(config, teams, parallel=True, show_progress=True)
ro.run(n_episode=1)