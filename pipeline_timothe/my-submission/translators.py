from pyrsistent import m
from utils import *
from typing import Dict
from abc import ABC, abstractmethod
import gym
import nmmo
from copy import deepcopy
from ScriptedAgents import AttackMageForTranslators
from ijcai2022nmmo.scripted import utils as utilsIJCAI


class ObservationToObservationUsefull(ABC):
    @abstractmethod
    def process(self, observation: Obs) -> Obs:
        """
        Example:
        >>> print(ObservationToObservationUsefull().process({'Entity': 1, 'Tile': 2})
        >>> {'Entity': 1}  #dans un gym.Space bien défini genre un Dict ou quoi
        """


class ActionUsefullToAction(ABC):
    @abstractmethod
    def traduce(self, obs: Obs, action: Action) -> Dict[type, Dict]:
        """
        Example:
        >>> print(ActionUsefullToAction().traduce(2))
        >>> {nmmo.action.Attack:   {nmmo.action.Style: style,
                                    nmmo.action.Target: targetID},
             nmmo.action.Move:      {nmmo.action.Direction: direction},
            }
        """


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


class MyObservationToObservationUsefull_v1(ObservationToObservationUsefull):
    def get_row_index(self, list, ID):
        for i, cand in enumerate(list):
            if cand == ID:
                # print(i)
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
                if power > BFF["Power"] and i != 0:
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

    def dist_and_lvl_diff(self, perso_info, entity_info):
        sr = perso_info[dict_feature_col["Entity"]["Continuous"]["Row_index"]]
        sc = perso_info[dict_feature_col["Entity"]
                        ["Continuous"]["Column_index"]]
        er = entity_info[dict_feature_col["Entity"]["Continuous"]["Row_index"]]
        ec = entity_info[dict_feature_col["Entity"]
                         ["Continuous"]["Column_index"]]
        dist = utilsIJCAI.l1((sr, sc), (er, ec))
        dist = np.array([dist])

        lvl_perso = perso_info[dict_feature_col["Entity"]
                               ["Continuous"]["Level"]]
        lvl_entity = entity_info[dict_feature_col["Entity"]
                                 ["Continuous"]["Level"]]
        lvl_diff = np.array([lvl_entity-lvl_perso])

        return dist, lvl_diff

    def preprocess_other_entity(self, info_cards, perso_all_info, observation_unmask, entity_idx, keep_col_other_ent):
        entity_all_info = observation_unmask["Entity"]["Continuous"][entity_idx, :]
        entity_info = entity_all_info[keep_col_other_ent]
        distance, lvl_diff = self.dist_and_lvl_diff(
            perso_all_info, entity_all_info)
        entity_info = np.concatenate(
            [entity_info, lvl_diff, distance], axis=-1)
        # late to assure the good shape
        if entity_idx == 0:
            entity_info = 0*entity_info
        info_cards.append(entity_info)

    def process(self, observation: Obs) -> Obs:
        '''
        Variable:
        -observation of a single soldier
        return:
        -observationUsefull (dict) = {
            'perso_info':array with health food water,
            'other_entities_info': array health is_freezed diff_lvl distance(l1),
            'N_ennemy_players':,
            'N_NPC':,
            'N_friends':,
        }
        '''
        # features to keep
        keep_col_perso = [dict_feature_col['Entity']['Continuous']
                          [feature] for feature in ['Health', 'Food', 'Water']]
        keep_col_other_ent = [dict_feature_col['Entity']['Continuous']
                              [feature] for feature in ['Health', 'Is_freezed', 'Level', 'Row_index', 'Column_index']]

        ################ extract where mask==0 #############
        # arr = arr[arr[:,0] == 6]
        observation_unmask = deepcopy(observation)

        observation_unmask["Entity"]["Continuous"] = observation_unmask["Entity"][
            "Continuous"][observation["Entity"]["Continuous"][:, 0] == 1]
        # what we want, only to not drift away
        observationUsefull = {
            "perso_info": None,
            "other_entities_info": None,
            'N_ennemy_players': None,
            'N_NPCs': None,
            'N_friends': None,
        }

        #
        other_ent_info_cards = []
        # get info entity
        perso_all_info = observation_unmask["Entity"]["Continuous"][0, :]
        perso_info = perso_all_info[keep_col_perso]

        # add perso_info to observationUsefull
        observationUsefull["perso_info"] = perso_info
        # get info attacker
        Attacker_ID = observation["Entity"]["Continuous"][0,
                                                          dict_feature_col["Entity"]["Continuous"]["Attacker_ID"]]
        Attacker_idx = self.get_row_index(
            observation_unmask["Entity"]["Continuous"][:, dict_feature_col["Entity"]["Continuous"]["Entity_ID"]], Attacker_ID)

        self.preprocess_other_entity(
            other_ent_info_cards, perso_all_info, observation_unmask, Attacker_idx, keep_col_other_ent)

        # get info around
        N_friends, N_players, N_NPCs, BFF_idx, weakest_player_idx, weakest_NPC_idx = self.scan_soldiers_around(
            observation_unmask["Entity"]["Continuous"], observation_unmask["Entity"]["Continuous"][0, dict_feature_col["Entity"]["Continuous"]["Population"]])

        observationUsefull["N_friends"] = np.array([N_friends])
        observationUsefull["N_ennemy_players"] = np.array([N_players])
        observationUsefull["N_NPCs"] = np.array([N_NPCs])

        # if one among BFF, weakest_player, or weaker_NPC is not perceived, its info is set to a 0 matrix
        self.preprocess_other_entity(
            other_ent_info_cards, perso_all_info, observation_unmask, BFF_idx, keep_col_other_ent)
        self.preprocess_other_entity(
            other_ent_info_cards, perso_all_info, observation_unmask, weakest_player_idx, keep_col_other_ent)
        self.preprocess_other_entity(
            other_ent_info_cards, perso_all_info, observation_unmask, weakest_NPC_idx, keep_col_other_ent)

        entities_infos = np.stack([other_ent_info_cards[i]
                                   for i in range(len(other_ent_info_cards))], axis=0)
        observationUsefull["other_entities_info"] = entities_infos
        return observationUsefull


class MyActionUsefullToAction_AttackAgent_v1(ActionUsefullToAction):
    '''
    the call of the attackAgent must take observation and idx_action as variables
    '''

    def __init__(self, config, idx=0, class_attack_agent=AttackMageForTranslators):
        self.attack_agent = class_attack_agent(
            config, idx)  # the idx set here does not matter

    def traduce(self, observation: Obs, action: Action) -> Dict[type, Dict]:
        '''
        Variable:
        -action (int): integer in [-4,5] to globally order what to do
        Return:
        -res_action (dict) with values in NMMO.action.---: {
            nmmo.action.Attack:   {nmmo.action.Style: styleScripted,
                                   nmmo.action.Target: targetIDScripted},
            nmmo.action.Move:      {nmmo.action.Direction: directionTakenByAgent},
                    }
            to fulfill the order
        '''
        res_action = self.attack_agent(observation, action)
        return res_action


def build_translators(config, idx=0, class_attack_agent=AttackMageForTranslators):
    translators = {num_team: {num_soldier: {"obs2obsUsefull": MyObservationToObservationUsefull_v1(),
                                            "actionUsefull2action": MyActionUsefullToAction_AttackAgent_v1(config, idx=idx, class_attack_agent=class_attack_agent)}
                              for num_soldier in range(8)} for num_team in range(16)}
    return translators
