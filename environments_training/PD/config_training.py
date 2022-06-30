from ray.rllib.policy.policy import PolicySpec

from PD.env import PD_MultiAgentEnv      #to register PD_MultiAgentEnv in gym_spaces
from PD.policies import PrisonerPolicy
from PD.gym_spaces import prisoner_action_space, prisoner_observation_space
import PD.CONFIG as CONFIG


from typing import Dict 

def policy_map_fn(agent_id: str, _episode=None, _worker=None, **_kwargs) -> str:
    """
    Maps agent_id to policy_id
    """
    if "prisoner" in agent_id:
        return "prisoner_policy"
    else:
        raise ValueError("Unknown agent_id: {}".format(agent_id))

def get_multiagent_policies() -> Dict[str,PolicySpec]:
    policies: Dict[str,PolicySpec] = {}  # policy_id to policy_spec

    #Policy of the VirtualAgent "prisoner"
    policies['prisoner_policy'] = PolicySpec(
                policy_class=None, # use default in trainer, or could be YourHighLevelPolicy
                observation_space=prisoner_observation_space,
                action_space=prisoner_action_space,
                config={}
    )

    return policies


policies = get_multiagent_policies()

# see https://github.com/ray-project/ray/blob/releases/1.10.0/rllib/agents/trainer.py
config_concerning_training = {
        "num_workers" : 2,           # <!> 0 for single-machine training
        "simple_optimizer": True,
        "ignore_worker_failures": True,
        "batch_mode": "complete_episodes",
        "framework": "torch",
        "model" : {
            "fcnet_hiddens": [1],
            "fcnet_activation": "linear",
            },
        # "env": "multiagent_PrisonerDilemma",          #other way to specify env. Requires to have register the env before.
        "env": PD_MultiAgentEnv,
        "env_config": {"n_players": CONFIG.n_players},

        ### MultiAgent config : defines (training?) policies and virtual_agent_id to policy_id mapping.
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_map_fn,
            "policies_to_train": list(policies.keys()),

            "count_steps_by": "env_steps",
            "observation_fn": None,
            "replay_mode": "independent",
            "policy_map_cache": None,
            "policy_map_capacity": 100,
        },
        
        ### Evaluation config
        "evaluation_num_workers": 1,
        "evaluation_config": {
            "render_env": True,
            }
    }
