from ijcai2022nmmo.scripted.baselines import Scripted
from ijcai2022nmmo.scripted.scripted_team import ScriptedTeam
import nmmo


class AttackMage(Scripted):
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
    agent_klass = AttackMage
