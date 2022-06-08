from pathlib import Path

from ijcai2022nmmo import CompetitionConfig, RollOut, scripted

from submission import MonobeastBaselineTeam


def rollout():
    config = CompetitionConfig()
    checkpoint_path = Path(
        __file__).parent / "checkpoints" / "model_113016832.pt"
    my_team = MonobeastBaselineTeam(team_id="my_team",
                                    env_config=config,
                                    checkpoint_path=checkpoint_path)
    all_teams = [scripted.CombatTeam(f"combat-{i}", config) for i in range(3)] + \
        [scripted.ForageTeam(f"forage-{i}", config) for i in range(5)] + \
        [scripted.RandomTeam(f"random-{i}", config) for i in range(7)] + \
        [my_team]
    ro = RollOut(config, all_teams, True)
    ro.run(n_episode=1, render=False)


if __name__ == "__main__":
    rollout()
