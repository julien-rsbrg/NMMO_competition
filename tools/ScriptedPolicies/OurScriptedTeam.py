from ijcai2022nmmo.scripted.baselines import Scripted
from ijcai2022nmmo.scripted.scripted_team import ScriptedTeam
from ijcai2022nmmo.scripted.move import evade as move_evade
import nmmo
from nmmo.io import action
from ijcai2022nmmo import Team
import numpy as np

# To redo ScriptedTeam
from typing import Dict, Type, List

from ijcai2022nmmo.evaluation.team import Team
from ijcai2022nmmo.scripted import baselines
'''
Comments: the only thing to master is the style of attack. A player will always want to kill an ennemy and his movements will be controlled by RL.
'''

#######################################################


class NotMovingTeam(Team):
    def __init__(self, team_id: str, config=None, **kwargs):
        super().__init__(team_id, config)

    def act(self, observations):
        actions = {}
        for player_idx, obs in observations.items():
            actions[player_idx] = {}
        return actions

# WIP


#######################################################
class GoNorthTeam(Team):
    def __init__(self, team_id: str, config=None, **kwargs):
        super().__init__(team_id, config)

    def act(self, observations):
        actions = {}
        for player_idx, obs in observations.items():
            actions[player_idx] = {}
        return actions

#######################################################


class CombatMage(Scripted):
    name = 'CombatMag_'
    '''Forages, fights, and explores.

    Uses a slightly more sophisticated attack routine
    Uses Mage style only
    '''

    def __call__(self, obs):
        super().__call__(obs)

        self.adaptive_control_and_targeting()

        self.style = nmmo.action.Mage
        self.attack()

        return self.actions


class CombatMageTeam(ScriptedTeam):
    agent_klass = CombatMage

#######################################################


class CombatMR(Scripted):
    name = 'CombatMR_'
    '''Forages, fights, and explores.

    Uses a slightly more sophisticated attack routine
    Uses ratio Mage Range
    '''

    def __init__(self, config, idx, epsilon_MR):
        '''
        Args:
           config : A forge.blade.core.Config object or subclass object
        '''
        super().__init__(config, idx)
        self.epsilon_MR = epsilon_MR

    def __call__(self, obs):
        super().__call__(obs)

        self.adaptive_control_and_targeting()
        style_criterion = np.random.uniform(0, 1,)
        if style_criterion <= self.epsilon_MR:
            self.style = nmmo.action.Mage
        else:
            self.style = nmmo.action.Range
        self.attack()

        return self.actions


class CombatMRTeam(Team):
    agent_klass = CombatMR
    agents: List[baselines.Scripted]

    def __init__(
        self,
        team_id: str,
        env_config: nmmo.config.Config,
        epsilon_MR: float,
        **kwargs,
    ) -> None:
        if "policy_id" not in kwargs:
            kwargs["policy_id"] = self.agent_klass.__name__
        super().__init__(team_id, env_config, **kwargs)
        self.epsilon_MR = epsilon_MR
        self.reset()

    def reset(self):
        assert self.agent_klass
        self.agents = [
            self.agent_klass(self.env_config, i, self.epsilon_MR)
            for i in range(self.env_config.TEAM_SIZE)
        ]

    def act(self, observations: Dict[int, dict]) -> Dict[int, dict]:
        actions = {i: self.agents[i](obs) for i, obs in observations.items()}
        for i in actions:
            for atn, args in actions[i].items():
                for arg, val in args.items():
                    if len(arg.edges) > 0:
                        actions[i][atn][arg] = arg.edges.index(val)
                    else:
                        targets = self.agents[i].targets
                        actions[i][atn][arg] = targets.index(val)
        return actions

#######################################################


class CombatTribrid1(Scripted):
    name = 'CombatTri1_'
    '''Forages, fights, and explores.

    Uses a slightly more sophisticated attack routine'''

    def __init__(self, config, idx):
        super().__init__(config, idx)
        self.evading = False

    def evade(self):
        '''Target and path away from an attacker'''
        move_evade(self.config, self.ob, self.actions, self.attacker)
        self.target = self.attacker
        self.targetID = self.attackerID
        self.targetDist = self.attackerDist

    def select_combat_style(self, obs):
        '''Select a combat style based on distance from the current target'''
        if self.target is None:
            return

        if self.targetDist <= self.config.COMBAT_MELEE_REACH:
            self.style = nmmo.action.Melee
        elif self.targetDist <= self.config.COMBAT_RANGE_REACH:
            self.style = nmmo.action.Range
        # and obs["Health"] > 4
        elif self.targetDist <= self.config.COMBAT_MAGE_REACH:
            self.style = nmmo.action.Mage
        # elif self.evading and self.targetDist <= self.config.COMBAT_MAGE_REACH and obs["Health"]>4: #and self.hp low
        #     self.style = nmmo.action.Mage
        else:
            self.style = nmmo.action.Range

    def adaptive_control_and_targeting(self, explore=True):
        '''Balanced foraging, evasion, and exploration'''
        self.scan_agents()

        if self.attacker is not None:
            self.evade()
            self.evading = True
            return

        self.evading = False
        if self.forage_criterion or not explore:
            self.forage()
        else:
            self.explore()

        self.target_weak()

    def __call__(self, obs):
        super().__call__(obs)

        # balance forage, attack, evade, explore
        # todo: evade to reconsider
        self.adaptive_control_and_targeting(obs)

        # todo: consider nmmo.Action.Mage style when evading or freezing
        self.select_combat_style(obs)
        self.attack()
        return self.actions


class CombatTribrid1Team(ScriptedTeam):
    agent_klass = CombatTribrid1
