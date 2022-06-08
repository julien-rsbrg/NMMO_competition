from ijcai2022nmmo import CompetitionConfig, scripted, submission, RollOut

config = CompetitionConfig()


my_team = submission.get_team_from_submission(
    submission_path="nmmo-v1/my-submission/",
    team_id="MyTeam",
    env_config=config,
)
# Or initialize the team directly
# my_team = MyTeam("Myteam", config, ...)

teams = []
teams.extend([scripted.CombatTeam(f"Combat-{i}", config) for i in range(3)])
teams.extend([scripted.ForageTeam(f"Forage-{i}", config) for i in range(5)])
teams.extend([scripted.RandomTeam(f"Random-{i}", config) for i in range(7)])
teams.append(my_team)

ro = RollOut(config, teams, parallel=True, show_progress=True)
# ro.run(n_episode=1)
print('config:', config)
