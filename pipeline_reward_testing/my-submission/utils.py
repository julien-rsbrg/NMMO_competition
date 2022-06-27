import numpy as np

def obs_dict_to_string(obs_dict):
    if isinstance(obs_dict, dict):
        L = [str(key) + ": " + obs_dict_to_string(value) for key, value in obs_dict.items()]
        return "(" + ")(".join(L) + ")"
    elif isinstance(obs_dict, np.ndarray):
        return "ndarray[" + str(obs_dict.shape) + "]" 
    else: 
        raise Exception("obs_dict_to_string: obs_dict is not a dict or a numpy.ndarray")
    
class Obs: pass
class Action: pass
class AgentID(str): pass