from ijcai2022nmmo.scripted.baselines import Scripted
import nmmo


class AttackMageForTranslators(Scripted):
    name = 'CombatMag_'
    '''Forages, fights, and explores.

    Uses a slightly more sophisticated attack routine
    Uses Mage style only
    '''

    def __call__(self, obs):
        self.scan_agents()
        self.target_weak()
        self.style = nmmo.action.Mage
        self.attack()
        return self.actions
