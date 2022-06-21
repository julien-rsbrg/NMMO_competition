from ray.rllib.policy.policy import PolicySpec

import CartPole.env #to register multiagent_CartPole-v1 as a MA env
# from CartPole.gym_spaces import *
import CartPole.CONFIG as CONFIG


from typing import Dict 

def policy_map_fn(agent_id: str, _episode=None, _worker=None, **_kwargs) -> str:
    """
    Maps agent_id to policy_id
    """
    return "agent_policy"
    if "agent" in agent_id:
        return "agent_policy"
    else:
        raise ValueError("Unknown agent_id: {}".format(agent_id))

def get_multiagent_policies() -> Dict[str,PolicySpec]:
    policies: Dict[str,PolicySpec] = {}  # policy_id to policy_spec
    #Policy of the VirtualAgent "prisoner"
    policies['agent_policy'] = PolicySpec(
                policy_class=None, # use default in trainer, or could be YourHighLevelPolicy
                observation_space=None,
                action_space=None,
                config={}
    )
    return policies


policies = get_multiagent_policies()

# see https://github.com/ray-project/ray/blob/releases/1.10.0/rllib/agents/trainer.py
config_concerning_training = {
        "num_workers" : 0,           # <!> 0 for single-machine training
        "simple_optimizer": True,
        "ignore_worker_failures": True,
        "batch_mode": "complete_episodes",
        "framework": "torch",

        # "env": "multiagent_PrisonerDilemma",          #other way to specify env. Requires to have register the env before.
        "env": "multiagent_CartPole-v1",
        "env_config": {"num_agents": CONFIG.num_agents},

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
    }
