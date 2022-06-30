import nmmo
import openskill
from tqdm import tqdm
from loguru import logger
from concurrent import futures
from typing import Dict, List, Optional

from ijcai2022nmmo.evaluation.rating import RatingSystem
from ijcai2022nmmo.evaluation.team import Team
from ijcai2022nmmo.env.team_based_env import TeamBasedEnv
from ijcai2022nmmo.env.metrics import Metrics
from ijcai2022nmmo.timer import timer
from ijcai2022nmmo.evaluation import analyzer
from ijcai2022nmmo.exception import TeamTimeoutError


class RollOut(object):
    env: TeamBasedEnv
    teams: List[Team]
    parallel: bool
    show_progress: bool
    _executor: Optional[futures.ThreadPoolExecutor] = None

    def __init__(
        self,
        env_config: nmmo.config.Config,
        teams: List[Team],
        parallel: bool = False,
        show_progress: bool = True,
    ) -> None:
        assert len(
            teams
        ) == env_config.NPOP, f"number of teams ({len(teams)}) is different with config ({env_config.NPOP})"
        assert len(env_config.AGENTS) == env_config.NPOP

        # use team.id as agent.name, so players can be identified in replay
        for i, team in enumerate(teams):

            class Agent(nmmo.Agent):
                name = f"{team.id}_"

            env_config.AGENTS[i] = Agent

        self.env = TeamBasedEnv(env_config)
        self.teams = teams
        self.parallel = parallel
        if parallel:
            self._executor = futures.ThreadPoolExecutor(max_workers=len(teams))
        self.show_progress = show_progress

    def reset(self) -> Dict[int, Dict[int, dict]]:
        observations_by_team = self.env.reset()
        if self.parallel:
            fs: Dict[int, futures.Future] = {}
            for team_idx, team in enumerate(self.teams):
                fs[team_idx] = self._executor.submit(team.reset)
            for team_idx, f in fs.items():
                try:
                    f.result()
                except Exception as e:
                    logger.warning(e)
        else:
            for team in self.teams:
                team.reset()
                team.n_timeout = 0
        return observations_by_team

    def run(
        self,
        n_timestep: int = 1024,
        n_episode: int = 1,
        render: bool = False,
    ) -> List[Dict[int, analyzer.TeamResult]]:
        results = []
        episode = 0
        while episode < n_episode:
            episode += 1
            print(f"Start Episode {episode}!")

            if self.show_progress:
                pbar = tqdm(total=n_timestep)

            observations_by_team = self.reset()

            timestep = 0
            while observations_by_team and timestep < n_timestep:
                if self.show_progress:
                    pbar.update()

                if render:
                    self.env.render()

                with timer.count("get_actions"):
                    actions_by_team = self._get_actions_by_team(
                        observations_by_team)

                with timer.count("env.step"):
                    observations_by_team, _, _, _ = self.env.step(
                        actions_by_team)

                timestep += 1

            if self.show_progress:
                pbar.close()

            self._end_episode(episode, results)

        return self._end_run(results)

    def _print(self,
               result_by_team: Dict[int, analyzer.TeamResult],
               ratings: Optional[Dict[int, openskill.Rating]] = None,
               topn_probs: Optional[Dict[int, float]] = None,
               n: int = 1):
        metrics_keys = Metrics.names()

        import prettytable
        from prettytable import PrettyTable
        table = PrettyTable()
        field_names = ["Team"] + metrics_keys + ["Achievement"]
        if ratings:
            field_names += ["Rating"]
        if topn_probs:
            field_names += [f"Top{n}Ratio"]

        table.field_names = field_names
        table.hrules = prettytable.ALL
        table.vrules = prettytable.FRAME
        for name in field_names:
            table.align[name] = "l"
        for team_idx, result in result_by_team.items():
            stat = result.stat

            row = [
                f"{self.teams[team_idx].id}\n({self.teams[team_idx].policy_id})"
            ] + [
                f"max: {stat['max'][key]:.2f}\navg: {stat['avg'][key]:.2f}"
                for key in metrics_keys
            ]
            row += [f"{result.achievement:.2f}"]
            if ratings:
                row += [f"{ratings[team_idx].mu:.2f}"]
            if topn_probs:
                row += [f"{topn_probs[team_idx]:.2f}"]
            table.add_row(row)
        print(table)

    def _get_actions_by_team(
            self, observations_by_team) -> Dict[int, Dict[int, dict]]:
        actions_by_team: Dict[int, Dict[int, dict]] = {}

        if self.parallel:
            fs: Dict[int, futures.Future] = {}
            for team_idx, observations in observations_by_team.items():
                team = self.teams[team_idx]
                fs[team_idx] = self._executor.submit(team.act, observations)
            for team_idx, f in fs.items():
                try:
                    actions_by_team[team_idx] = f.result()
                except TeamTimeoutError as e:
                    logger.warning(e)
                    self.teams[team_idx].n_timeout += 1
                except Exception as e:
                    logger.warning(e)
        else:
            for team_idx, observations in observations_by_team.items():
                team = self.teams[team_idx]
                actions_by_team[team_idx] = team.act(observations)

        return actions_by_team

    def _end_episode(self, episode: int,
                     results: List[Dict[int, analyzer.TeamResult]]):
        with timer.count("env.terminal"):
            self.env.terminal()

        stat_by_team = self.env.stat_by_team()
        results.append({
            i: analyzer.TeamResult(self.teams[i].policy_id, stat,
                                   self.teams[i].n_timeout)
            for i, stat in stat_by_team.items()
        })

        print(f"Result of Episode {episode}:")
        self._print(results[-1])

    def _end_run(self, results: List[Dict[int, analyzer.TeamResult]]):
        ret = results

        rs = RatingSystem(self.teams)
        for result_by_team in results:
            rs.update(
                self.teams,
                [result.achievement for result in result_by_team.values()],
            )

        results: Dict[int, analyzer.TeamResult] = analyzer.avg_results(results)
        ratings = rs.get_team_ratings(self.teams)

        print("Final Result:")
        self._print(
            results,
            # ratings,
            topn_probs=analyzer.topn_probs(ret),
        )
        for team_idx, result in results.items():
            print('team_idx:', team_idx,
                  "## result.achievement:", result.achievement)
        return ret


if __name__ == "__main__":
    from ijcai2022nmmo import CompetitionConfig
    config = CompetitionConfig()

    from ijcai2022nmmo import scripted
    teams = [scripted.CombatTeam(f"MyTeam", config, policy_id="My")]
    teams.extend(
        [scripted.CombatTeam(f"Combat-{i}", config) for i in range(3)])
    teams.extend(
        [scripted.ForageTeam(f"Forage-{i}", config) for i in range(3)])
    teams.extend(
        [scripted.RandomTeam(f"Random-{i}", config) for i in range(3)])
    teams.extend([
        scripted.CombatNoExploreTeam(f"CombatNoExplore-{i}", config)
        for i in range(3)
    ])
    teams.extend([
        scripted.ForageNoExploreTeam(f"ForageNoExplore-{i}", config)
        for i in range(3)
    ])
    ro = RollOut(
        config,
        teams,
        show_progress=True,
    )
    ro.run(n_timestep=20, n_episode=2)
