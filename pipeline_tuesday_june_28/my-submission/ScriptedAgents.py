from ijcai2022nmmo.scripted.baselines import Scripted
import nmmo
from nmmo import scripting
from ijcai2022nmmo.scripted import move
from ijcai2022nmmo.scripted import utils as utilsIJCAI
from copy import deepcopy

entity_cont_features = ['Mask', "Entity_ID", "Attacker_ID", "Level", "Population",
                        "Row_index", "Column_index", "Damage", "Timealive", "Food", "Water", "Health", "Is_freezed"]
entity_discrete_features = ['Mask', "Population", "Row_index", "Column_index"]
tile_cont_features = ["N_entity", "Type", "Row_index", "Column_index"]
tile_discrete_features = ["Type", "Row_index", "Column_index"]
dict_feature_col = {
    "Entity": {
        "Continuous": {feature: i for i, feature in enumerate(entity_cont_features)},
        "Discrete": {feature: i for i, feature in enumerate(entity_discrete_features)},
    },
    "Tile": {
        "Continuous": {feature: i for i, feature in enumerate(tile_cont_features)},
        "Discrete": {feature: i for i, feature in enumerate(tile_discrete_features)}
    }
}


class AttackMageForTranslators(Scripted):
    name = 'CombatMag_'
    '''Forages, fights, and explores.

    Uses a slightly more sophisticated attack routine
    Uses Mage style only
    '''

    def get_mtrx_row_index(self, list, ID):
        for i, cand in enumerate(list):
            if cand == ID:
                print(i)
                return i
        return 0

    def scan_soldiers_around(self, soldiers_caracteristics, own_population):
        def update_dic(dic, row, power):
            dic["Row_index"] = row
            dic["Power"] = power

        # if no ennemy its own one
        weakest_player = {"Row_index": 0, "Power": 0}
        N_players = 0

        weakest_NPC = {"Row_index": 0, "Power": 0}
        N_NPCs = 0

        BFF = {"Row_index": 0, "Power": 0}
        N_friends = 0
        for i in range(soldiers_caracteristics.shape[0]):
            cand_pop = soldiers_caracteristics[i,
                                               dict_feature_col["Entity"]["Continuous"]["Population"]]
            lvl = soldiers_caracteristics[i,
                                          dict_feature_col["Entity"]["Continuous"]["Level"]]
            hp = soldiers_caracteristics[i,
                                         dict_feature_col["Entity"]["Continuous"]["Health"]]
            dmg = soldiers_caracteristics[i,
                                          dict_feature_col["Entity"]["Continuous"]["Damage"]]
            power = lvl*2+hp-dmg
            if cand_pop == own_population:
                N_friends += 1
                if power > BFF["Power"]:
                    update_dic(BFF, i, power)
            elif cand_pop < 0:  # NPC
                N_NPCs += 1
                if power > weakest_NPC["Power"]:
                    update_dic(weakest_NPC, i, power)
            else:  # Player
                N_players += 1
                if power > weakest_player["Power"]:
                    update_dic(weakest_player, i, power)
        return N_friends, N_players, N_NPCs, BFF["Row_index"], weakest_player["Row_index"], weakest_NPC["Row_index"]

    def get_close(self, arr_entity_to_track):
        '''Only path to entity to track'''
        Entity = nmmo.Serialized.Entity

        sr = nmmo.scripting.Observation.attribute(self.ob.agent, Entity.R)
        sc = nmmo.scripting.Observation.attribute(self.ob.agent, Entity.C)

        gr = nmmo.scripting.Observation.attribute(
            arr_entity_to_track, Entity.R)
        gc = nmmo.scripting.Observation.attribute(
            arr_entity_to_track, Entity.C)

        rr, cc = (2 * sr + gr, 2 * sc + gc)

        move.pathfind(self.config, self.ob, self.actions, rr, cc)

    def evade(self, arr_entity_to_evade):
        '''Target and path away from an attacker'''
        move.evade(self.config, self.ob, self.actions, arr_entity_to_evade)

        Entity = nmmo.Serialized.Entity
        sr = nmmo.scripting.Observation.attribute(self.ob.agent, Entity.R)
        sc = nmmo.scripting.Observation.attribute(self.ob.agent, Entity.C)
        # this method could be broaden for our purpose

        tr = arr_entity_to_evade[dict_feature_col["Entity"]
                                 ['Continuous']['Row_index']]
        tc = arr_entity_to_evade[dict_feature_col["Entity"]
                                 ['Continuous']['Column_index']]
        # or
        # tr = nmmo.scripting.Observation.attribute(arr_entity_to_evade, Entity.R)
        # tc = nmmo.scripting.Observation.attribute(arr_entity_to_evade, Entity.C)

        self.target = arr_entity_to_evade
        self.targetID = nmmo.scripting.Observation.attribute(arr_entity_to_evade,
                                                             Entity.ID)
        self.targetDist = utilsIJCAI.l1((sr, sc), (tr, tc))

    def postprocess_actions(self, actions):
        '''
        because of an unexpected incompatibility
        '''
        dict_direction_to_idx = {item: i for i, item in enumerate(
            [nmmo.action.North, nmmo.action.South, nmmo.action.East, nmmo.action.West])}
        new_actions = {
            nmmo.action.Attack: {
                nmmo.action.Style: actions[nmmo.action.Attack][nmmo.action.Style].index,
                nmmo.action.Target: int(
                    actions[nmmo.action.Attack][nmmo.action.Target])
            },
            nmmo.action.Move: {
                nmmo.action.Direction: dict_direction_to_idx[actions[nmmo.action.Move]
                                                             [nmmo.action.Direction]]
            }
        }
        print('new_actions:', new_actions)
        return new_actions

    def precall(self, obs):
        '''
        should work as super.__call__(obs) minus a bug
        Process observations and return actions

        Args:
           obs: An observation object from the environment. Unpack with scripting.Observation
        '''
        self.actions = {}

        self.ob = scripting.Observation(self.config, obs)
        agent = self.ob.agent

        self.food = scripting.Observation.attribute(
            agent, nmmo.Serialized.Entity.Food)
        self.water = scripting.Observation.attribute(
            agent, nmmo.Serialized.Entity.Water)

        if self.food > self.food_max:
            self.food_max = self.food
        if self.water > self.water_max:
            self.water_max = self.water

        if self.spawnR is None:
            self.spawnR = scripting.Observation.attribute(
                agent, nmmo.Serialized.Entity.R)
        if self.spawnC is None:
            self.spawnC = scripting.Observation.attribute(
                agent, nmmo.Serialized.Entity.C)

    def __call__(self, obs, idx_action):
        self.precall(obs)

        # begin to handle attack
        self.scan_agents()
        # if you do not evade and that the closest character is weak enough, it will attack the latter
        self.target_weak()
        # could be tuned with nmmo.action.Range
        self.style = nmmo.action.Mage

        # get rid of the masked observations
        unmasked_obs = deepcopy(obs)
        arr_of_interest = unmasked_obs["Entity"][
            "Continuous"][obs["Entity"]["Continuous"][:, 0] == 1]

        # movement goal is set by the RL agent through its idx_action output
        if idx_action == 0:
            # explore
            self.explore()
        elif idx_action == 1:
            # forage
            self.forage()
        elif idx_action == 2 or idx_action == 3:
            # relates to attacker
            Attacker_ID = arr_of_interest[0,
                                          dict_feature_col["Entity"]["Continuous"]["Attacker_ID"]]
            Attacker_mtrx_row_index = self.get_mtrx_row_index(
                arr_of_interest[:, dict_feature_col["Entity"]["Continuous"]["Entity_ID"]], Attacker_ID)
            Attacker_info = arr_of_interest[Attacker_mtrx_row_index, :]
            if idx_action == 2:
                # move to attacker
                self.get_close(Attacker_info)
            elif idx_action == 3:
                # evade attacker
                # the same as the evade method by default
                self.evade(Attacker_info)
        elif idx_action in [4, 5, 6, 7, 8]:
            # relates to BFF, weakest ennemy player or NPC
            own_population = arr_of_interest[0,
                                             dict_feature_col["Entity"]["Continuous"]["Population"]]
            # or
            # own_population = nmmo.scripting.Observation.attribute(
            #     self.ob.agent, nmmo.Serialized.Entity.Population)
            _, _, _, BFF_mtrx_row_idx, weakest_ennemy_plr_mtrx_row_idx, weakest_NPC_mtrx_row_idx = self.scan_soldiers_around(
                arr_of_interest, own_population)
            if idx_action == 4:
                # move to BFF
                if BFF_mtrx_row_idx != 0:
                    BFF_info = arr_of_interest[BFF_mtrx_row_idx, :]
                    self.get_close(BFF_info)
                else:
                    self.explore()
            elif idx_action == 5:
                # move to weakest ennemy player
                if weakest_ennemy_plr_mtrx_row_idx != 0:
                    weakest_ennemy_plr_info = arr_of_interest[weakest_ennemy_plr_mtrx_row_idx, :]
                    self.get_close(weakest_ennemy_plr_info)
                else:
                    self.explore()
            elif idx_action == 6:
                # move to weakest NPC
                if weakest_NPC_mtrx_row_idx != 0:
                    weakest_NPC_info = arr_of_interest[weakest_NPC_mtrx_row_idx, :]
                    self.get_close(weakest_NPC_info)
                else:
                    self.explore()
            elif idx_action == 7:
                # evade weakest ennemy player
                if weakest_ennemy_plr_mtrx_row_idx != 0:
                    weakest_ennemy_plr_info = arr_of_interest[weakest_ennemy_plr_mtrx_row_idx, :]
                    self.evade(weakest_ennemy_plr_info)
                else:
                    self.explore()
            elif idx_action == 8:
                # evade weakest NPC
                if weakest_NPC_mtrx_row_idx != 0:
                    weakest_NPC_info = arr_of_interest[weakest_NPC_mtrx_row_idx, :]
                    self.evade(weakest_NPC_info)
                else:
                    self.explore()

        self.attack()
        self.actions = self.postprocess_actions(self.actions)
        return self.actions
