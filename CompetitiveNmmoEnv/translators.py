from utils import *
from typing import Dict
from abc import ABC, abstractmethod
import gym
import nmmo


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


class ObservationToZero(ObservationToObservationUsefull):
    def process(self, observation: Obs) -> Obs:
        return 0


class ObservationToObservationUsefull(ObservationToObservationUsefull):
    def get_row_index(self, list, ID):
        found = False
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

    def process(self, observation: Obs) -> Obs:
        '''
        Variable:
        -observation of a single soldier
        return:
        -observationUsefull (dict) = {
            "Entity":{
                "N_friends":, # kept in check among the 100 players the observations allow
                "N_NPCs":,
                "N_players":,
                "entities_info":, # array (4,11) infos of the given soldier, the attacker of the soldier, the best ally, the weakest player, the weakest NPC
            }
            "Tile":Tile_info #keep Type of the tiles only
        }
        '''
        # features to keep
        keep_col_til = list(range(1))  # "Type" from "Tile" "Discrete"
        keep_col_ent = list(range(1, 8))+list(range(9, 13))

        ################ extract where mask==0 #############
        # arr = arr[arr[:,0] == 6]
        observation["Entity"]["Continuous"] = observation["Entity"][
            "Continuous"][observation["Entity"]["Continuous"][:, 0] == 1]

        # initialize
        observationUsefull = {"Entity": {}, "Tile": {}}
        # get info Tile
        Tile_info = observation["Tile"]["Discrete"][:,
                                                    keep_col_til]  # !! list
        observationUsefull["Tile"] = Tile_info

        #
        info_cards = []
        # get info entity
        Entity_info = observation["Entity"]["Continuous"][0, keep_col_ent]
        info_cards.append(Entity_info)
        # get info attacker
        Attacker_ID = observation["Entity"]["Continuous"][0,
                                                          dict_feature_col["Entity"]["Continuous"]["Attacker_ID"]]
        Attacker_ID_row_index = self.get_row_index(
            observation["Entity"]["Continuous"][:, dict_feature_col["Entity"]["Continuous"]["Entity_ID"]], Attacker_ID)
        Attacker_info = observation["Entity"]["Continuous"][Attacker_ID_row_index, keep_col_ent]
        if Attacker_ID_row_index == 0:
            Attacker_info = 0*Attacker_info
        info_cards.append(Attacker_info)
        # get info around
        N_friends, N_players, N_NPCs, BFF_idx, weakest_player_idx, weakest_NPC_idx = self.scan_soldiers_around(
            observation["Entity"]["Continuous"], observation["Entity"]["Continuous"][0, dict_feature_col["Entity"]["Continuous"]["Population"]])

        observationUsefull["Entity"]["N_friends"] = N_friends
        observationUsefull["Entity"]["N_players"] = N_players
        observationUsefull["Entity"]["N_NPCs"] = N_NPCs

        # if one among BFF, weakest_player, or weaker_NPC is not perceived, its info is set to a 0 matrix
        BFF_info = observation["Entity"]["Continuous"][BFF_idx, keep_col_ent]
        if BFF_idx == 0:
            BFF_info = 0*BFF_info
        info_cards.append(BFF_info)
        weakest_player_info = observation["Entity"]["Continuous"][weakest_player_idx, keep_col_ent]
        if weakest_player_idx == 0:
            weakest_player_info = 0*weakest_player_info
        info_cards.append(weakest_player_info)
        weakest_NPC_info = observation["Entity"]["Continuous"][weakest_NPC_idx, keep_col_ent]
        if weakest_NPC_idx == 0:
            weakest_NPC_info = 0*weakest_NPC_info
        info_cards.append(weakest_NPC_info)

        entities_infos = np.stack([info_cards[i]
                                   for i in range(len(info_cards))], axis=0)
        observationUsefull["Entity"]["entities_infos"] = entities_infos
        return observationUsefull


class ActionUsefullToNoAction(ActionUsefullToAction):
    def traduce(self, action: Action) -> Dict[type, Dict]:
        return {}


class ActionUsefullToAction(ActionUsefullToAction):

    def traduce(self, action: Action) -> Dict[type, Dict]:
        '''
        Variable:
        -action (dict): {
            "attack":{
                'style':, # (in Discrete(3))
                'targetID':, # (in Discrete(129))
                },
            "move":{
                "direction":
            }
            }
        Return:
        ...
        '''
        styleUsefull = action["attack"]['style']  # (in Discrete(3)) even if scripted
        style = nmmo.action.Style.edges[styleUsefull]

        targetID = action["attack"]["targetID"]  # (in Discrete(129))

        directionUsefull = action["move"]['direction']  # (in Discrete(5))
        if directionUsefull == 4:
            # do not move
            return {nmmo.action.Attack:   {nmmo.action.Style: style,
                                           nmmo.action.Target: targetID},
                    nmmo.action.Move:      {nmmo.action.Direction: (0, 0)},
                    }
        else:
            direction = nmmo.action.Direction.edges[directionUsefull]
            return {nmmo.action.Attack:   {nmmo.action.Style: style,
                                           nmmo.action.Target: targetID},
                    nmmo.action.Move:      {nmmo.action.Direction: direction},
                    }
