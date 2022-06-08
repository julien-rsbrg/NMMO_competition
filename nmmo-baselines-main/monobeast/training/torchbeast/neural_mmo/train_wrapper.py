import nmmo
import numpy as np
from gym import Wrapper, spaces
from ijcai2022nmmo import TeamBasedEnv
from ijcai2022nmmo.scripted import CombatTeam, ForageTeam, RandomTeam
from ijcai2022nmmo.scripted.baselines import Scripted
from ijcai2022nmmo.scripted.scripted_team import ScriptedTeam


class FeatureParser:
    map_size = 15
    n_actions = 5
    NEIGHBOR = [(6, 7), (8, 7), (7, 8), (7, 6)]  # north, south, east, west
    OBSTACLE = (0, 1, 5)  # lava, water, stone
    feature_spec = {
        "terrain": spaces.Box(low=0, high=6, shape=(15, 15), dtype=np.int64),
        "camp": spaces.Box(low=0, high=4, shape=(15, 15), dtype=np.int64),
        "entity": spaces.Box(low=0,
                             high=4,
                             shape=(7, 15, 15),
                             dtype=np.float32),
        "va": spaces.Box(low=0, high=2, shape=(5, ), dtype=np.int64),
    }

    def parse(self, obs):
        ret = {}
        for agent_id in obs:
            terrain = np.zeros((self.map_size, self.map_size), dtype=np.int64)
            camp = np.zeros((self.map_size, self.map_size), dtype=np.int64)
            entity = np.zeros((7, self.map_size, self.map_size),
                              dtype=np.float32)
            va = np.ones(self.n_actions, dtype=np.int64)

            # terrian feature
            tile = obs[agent_id]["Tile"]["Continuous"]
            LT_R, LT_C = tile[0, 2], tile[0][3]
            for line in tile:
                terrain[int(line[2] - LT_R),
                        int(line[3] - LT_C)] = int(line[1])

            # npc and player
            raw_entity = obs[agent_id]["Entity"]["Continuous"]
            P = raw_entity[0, 4]
            _r, _c = raw_entity[0, 5:7]
            assert int(_r - LT_R) == int(
                _c - LT_C) == 7, f"({int(_r - LT_R)}, {int(_c - LT_C)})"
            for line in raw_entity:
                if line[0] != 1:
                    continue
                raw_pop, raw_r, raw_c = line[4:7]
                r, c = int(raw_r - LT_R), int(raw_c - LT_C)
                if camp[r, c] == 0:
                    continue
                camp[r, c] = 1 if raw_pop == P else np.sign(raw_pop) + 2
                # level
                entity[0, r, c] = line[3]
                # damage, timealive, food, water, health, is_freezed
                entity[1:, r, c] = line[7:]

            # valid action
            for i, (r, c) in enumerate(self.NEIGHBOR):
                if terrain[r, c] in self.OBSTACLE:
                    va[i + 1] = 0

            ret[agent_id] = {
                "terrain": terrain,
                "camp": camp,
                "entity": entity,
                "va": va
            }
        return ret


class RewardParser:

    def parse(self, prev_achv, achv):
        reward = {
            i: (sum(achv[i].values()) - sum(prev_achv[i].values())) / 100.0
            for i in achv
        }
        return reward


class TrainWrapper(Wrapper):
    max_step = 1024
    TT_ID = 0  # training team index
    use_auxiliary_script = False

    def __init__(self, env: TeamBasedEnv) -> None:
        super().__init__(env)
        self.feature_parser = FeatureParser()
        self.reward_parser = RewardParser()
        self.observation_space = spaces.Dict(self.feature_parser.feature_spec)
        self.action_space = spaces.Discrete(5)
        self._dummy_feature = {
            key: np.zeros(shape=val.shape, dtype=val.dtype)
            for key, val in self.observation_space.items()
        }

    def reset(self):
        raw_obs = super().reset()
        obs = raw_obs[self.TT_ID]
        obs = self.feature_parser.parse(obs)

        self.reset_auxiliary_script(self.config)
        self.reset_scripted_team(self.config)
        self.agents = list(obs.keys())
        self._prev_achv = self.metrices_by_team()[self.TT_ID]
        self._prev_raw_obs = raw_obs
        self._step = 0

        return obs

    def step(self, actions):
        decisions = self.get_scripted_team_decision(self._prev_raw_obs)
        decisions[self.TT_ID] = self.transform_action(
            actions,
            observations=self._prev_raw_obs[self.TT_ID],
            auxiliary_script=self.auxiliary_script)

        raw_obs, _, raw_done, raw_info = super().step(decisions)
        if self.TT_ID in raw_obs:
            obs = raw_obs[self.TT_ID]
            done = raw_done[self.TT_ID]
            info = raw_info[self.TT_ID]

            obs = self.feature_parser.parse(obs)
            achv = self.metrices_by_team()[self.TT_ID]
            reward = self.reward_parser.parse(self._prev_achv, achv)
            self._prev_achv = achv
        else:
            obs, reward, done, info = {}, {}, {}, {}

        for agent_id in self.agents:
            if agent_id not in obs:
                obs[agent_id] = self._dummy_feature
                reward[agent_id] = 0
                done[agent_id] = True

        self._prev_raw_obs = raw_obs
        self._step += 1

        if self._step >= self.max_step:
            done = {key: True for key in done.keys()}
        return obs, reward, done, info

    def reset_auxiliary_script(self, config):
        if not self.use_auxiliary_script:
            self.auxiliary_script = None
            return
        if getattr(self, "auxiliary_script", None) is not None:
            self.auxiliary_script.reset()
            return
        self.auxiliary_script = AttackTeam("auxiliary", config)

    def reset_scripted_team(self, config):
        if getattr(self, "_scripted_team", None) is not None:
            for team in self._scripted_team.values():
                team.reset()
            return
        self._scripted_team = {}
        assert config.NPOP == 16
        for i in range(config.NPOP):
            if i == self.TT_ID:
                continue
            if self.TT_ID < i <= self.TT_ID + 7:
                self._scripted_team[i] = RandomTeam(f"random-{i}", config)
            elif self.TT_ID + 7 < i <= self.TT_ID + 12:
                self._scripted_team[i] = ForageTeam(f"forage-{i}", config)
            elif self.TT_ID + 12 < i <= self.TT_ID + 15:
                self._scripted_team[i] = CombatTeam(f"combat-{i}", config)

    def get_scripted_team_decision(self, observations):
        decisions = {}
        tt_id = self.TT_ID
        for team_id, obs in observations.items():
            if team_id == tt_id:
                continue
            decisions[team_id] = self._scripted_team[team_id].act(obs)
        return decisions

    @staticmethod
    def transform_action(actions, observations=None, auxiliary_script=None):
        """neural network move + scripted attack"""
        decisions = {}

        # move decisions
        for agent_id, val in actions.items():
            if observations is not None and agent_id not in observations:
                continue
            if val == 0:
                decisions[agent_id] = {}
            elif 1 <= val <= 4:
                decisions[agent_id] = {
                    nmmo.action.Move: {
                        nmmo.action.Direction: val - 1
                    }
                }
            else:
                raise ValueError(f"invalid action: {val}")

        # attack decisions
        if auxiliary_script is not None:
            assert observations is not None
            attack_decisions = auxiliary_script.act(observations)
            # merge decisions
            for agent_id, d in decisions.items():
                d.update(attack_decisions[agent_id])
                decisions[agent_id] = d
        return decisions


class Attack(Scripted):
    '''attack'''
    name = 'Attack_'

    def __call__(self, obs):
        super().__call__(obs)

        self.scan_agents()
        self.target_weak()
        self.style = nmmo.action.Range
        self.attack()
        return self.actions


class AttackTeam(ScriptedTeam):
    agent_klass = Attack
