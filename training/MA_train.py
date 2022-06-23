"""
File for training agents on the Prisoner's Dilemma MA env.
"""
print("Imports...")
from typing import *
import shutil
import os

import ray
from ray.rllib.agents.ppo import ppo
from ray.rllib.policy.policy import PolicySpec
from gym import spaces

from utils import *



### CONFIGURATION ###
n_teams = 16
n_soldiers = 8
n_agents = n_teams * n_soldiers
RUN_WITH_TUNE = False
NUM_ITERATIONS = 100
CHECKPOINT_ROOT = "tmp/ppo/MA_train"

from CompetitiveEnv.env import CompetitionMultiAgentEnv
MA_ENV_CLASS = CompetitionMultiAgentEnv
MA_ENV_NAME = "CompetitionMultiAgentEnv"
MA_ENV_CONFIG = {"n_teams": n_teams, "n_soldiers": n_soldiers}

observation_space = spaces.Box(low = 0, high = CompetitionMultiAgentEnv.hp_max, shape = (n_agents,), dtype = np.float32)
action_space = spaces.Discrete(n_agents)





### TRAINER ###
config_concerning_trainer = ppo.DEFAULT_CONFIG.copy()
config_concerning_trainer.update(
    {
    # The batch size collected for each worker
    "rollout_fragment_length": 1000,
    # Can be "complete_episodes" or "truncate_episodes"
    "batch_mode": "complete_episodes",
    "simple_optimizer": True,
    })

class Trainer(ppo.PPOTrainer):
    pass







### MULTI AGENT CONFIG ###

def policy_map_fn(agent_id: str, _episode=None, _worker=None, **_kwargs) -> str:
    """
    Maps agent_id to policy_id
    """
    return "soldier_policy"

def get_multiagent_policies() -> Dict[str,PolicySpec]:
    policies: Dict[str,PolicySpec] = {}  # policy_id to policy_spec

    #Policy of the VirtualAgent "prisoner"
    policies['soldier_policy'] = PolicySpec(
                policy_class=None, # use default in trainer, or could be YourHighLevelPolicy
                observation_space=observation_space,
                action_space=action_space,
                config={}
    )
    return policies

policies = get_multiagent_policies()
policies_to_train = list(policies.keys())

multi_agent_config = {
                        "policies": policies,
                        "policy_mapping_fn": policy_map_fn,
                        "policies_to_train": policies_to_train,

                        "count_steps_by": "env_steps",
                        "observation_fn": None,
                        "replay_mode": "independent",
                        "policy_map_cache": None,
                        "policy_map_capacity": 100,
                    }





### CONFIG ###

config_concerning_training = {
        "num_workers" : 8,           # <!> 0 for single-machine training
        "simple_optimizer": True,
        "ignore_worker_failures": True,
        "batch_mode": "complete_episodes",
        "framework": "torch",
        "disable_env_checking": True,
        "model" : {
            "fcnet_hiddens": [8, 8],
            "fcnet_activation": "relu",
            },
        # "env": "multiagent_PrisonerDilemma",          #other way to specify env. Requires to have register the env before.
        "env": MA_ENV_NAME,
        "env_config": MA_ENV_CONFIG,

        ### MultiAgent config : defines (training?) policies and virtual_agent_id to policy_id mapping.
        "multiagent": multi_agent_config,
        
        ### Evaluation config
        "evaluation_num_workers": 1,
        "evaluation_config": {
            "render_env": True,
            }
    }

config = config_concerning_trainer.copy()
config.update(config_concerning_training)



###Initialize ray
print("Ray restarting...")
ray_init_info = ray.init(local_mode=True, ignore_reinit_error=True)  # in local mode you can debug it 
print("Dashboard URL: http://{}".format(ray_init_info["webui_url"]))



###CHECKPOINT
print("Defining checkpoint...")
#Crée le dossier checkpoint, cela va save des checkpoints de la policy dans ce dossier. 
#On pourra par la suite évaluer (faire des rollout) de cette policy avec l'agent (PPO) sur cet env (CartPole-v1) et pour 200 steps en tappant dans le cmd:
# rllib rollout CHECKPOINT_ROOT/checkpoint_000006/checkpoint-6 --config "{\"env\": \"multiagent_PrisonerDilemma\"}" --run PPO --steps 200
print("Define location for checkpoint...")
shutil.rmtree(CHECKPOINT_ROOT, ignore_errors=True, onerror=None)
ray_results = os.path.expanduser("~")  + "/ray_results/"



###TRAINING
print("Start training...")
trainer = ppo.PPOTrainer(config, env=MA_ENV_NAME)

s = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f} saved {}"
for n in range(NUM_ITERATIONS):
    result = trainer.train()
    #print(pretty_print(result))     #more like way_too_verbosy_print xd
    file_name = trainer.save(CHECKPOINT_ROOT)
    print(s.format(
        n + 1,
        result["episode_reward_min"],
        result["episode_reward_mean"],
        result["episode_reward_max"],
        result["episode_len_mean"],
        file_name
    ))



#DONE
print("Done.")