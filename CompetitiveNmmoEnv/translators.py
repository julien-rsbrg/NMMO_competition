from utils import *
from typing import Dict
from abc import ABC, abstractmethod


class ObservationToObservationUsefull(ABC):
    @abstractmethod
    def process(self, observation : Obs) -> Obs:
        """
        Example:
        >>> print(ObservationToObservationUsefull().process({'Entity': 1, 'Tile': 2})
        >>> {'Entity': 1}  #dans un gym.Space bien défini genre un Dict ou quoi
        """
    
class ActionUsefullToAction(ABC):
    @abstractmethod
    def traduce(self, action : Action) -> Dict[type, Dict]:
        """
        Example:
        >>> print(ActionUsefullToAction().traduce(2))
        >>> {nmmo.action.Attack:   {nmmo.action.Style: style,
                                    nmmo.action.Target: targetID},
             nmmo.action.Move:      {nmmo.action.Direction: direction},
            }
        """




class ObservationToObservation(ObservationToObservationUsefull):
    def process(self, observation : Obs) -> Obs:
        return observation
    
class ObservationToZero(ObservationToObservationUsefull):
    def process(self, observation : Obs) -> Obs:
        return 0
    
class ActionUsefullToNoAction(ActionUsefullToAction):
    def traduce(self, action : Action) -> Dict[type, Dict]:
        return {}