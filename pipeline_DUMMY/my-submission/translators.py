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
    def traduce(self, action: Action, **kwargs) -> Dict[type, Dict]:
        """
        Example:
        >>> print(ActionUsefullToAction().traduce(2))
        >>> {nmmo.action.Attack:   {nmmo.action.Style: style,
                                    nmmo.action.Target: targetID},
             nmmo.action.Move:      {nmmo.action.Direction: direction},
            }
        """

# TO CHANGE :
class MyObservationToObservationUsefull(ObservationToObservationUsefull):
    def process(self, observation: Obs, **kwargs) -> Obs:
        return 0

class MyActionUsefullToAction(ActionUsefullToAction):
    def traduce(self, action: Action, **kwargs) -> Dict[type, Dict]:
        return {}
    
translators = {num_team : {num_soldier : {"obs2obsUsefull" : MyObservationToObservationUsefull(),
                                                  "actionUsefull2action" : MyActionUsefullToAction()} 
    for num_soldier in range(8)} for num_team in range(16)}