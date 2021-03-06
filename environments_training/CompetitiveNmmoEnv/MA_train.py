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
from ray.rllib.agents.callbacks import DefaultCallbacks
from gym import spaces

from ijcai2022nmmo import CompetitionConfig

from utils import *
from env import soldier_action_space, soldier_observation_space


### CONFIGURATION ###
print("Configurating...")
n_teams = 16
n_soldiers = 8
n_agents = n_teams * n_soldiers
RUN_WITH_TUNE = False
do_render = True
NUM_ITERATIONS = 100
CHECKPOINT_ROOT = "tmp/ppo/NMMO_MA"

class ConfigTrain(CompetitionConfig):
    RENDER = False
class ConfigTest(CompetitionConfig):
    RENDER = do_render
MA_ENV_NAME = "CompetitionNmmoMultiAgentEnv"
MA_ENV_CONFIG = {"Config_class" : ConfigTrain}
MA_ENV_CONFIG_TEST = {"Config_class" : ConfigTest}

observation_space = soldier_observation_space
action_space = soldier_action_space





### TRAINER ###
config_concerning_trainer = ppo.DEFAULT_CONFIG.copy()
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
                        "policy_map_capacity": 200,
                        "count_steps_by": "env_steps",
                    }





### CONFIG ###
num_gpus = int(os.environ.get("RLLIB_NUM_GPUS", "0"))
print("Number of GPUs:", num_gpus)

config_concerning_training = {
        # === General settings ===
        "framework": "torch",
        "disable_env_checking": True,
        
        
        # === Worker & sampling settings ===
        "num_workers" : 5,           # <!> 0 for single-machine training. Number of rollout worker actors to create for parallel sampling
        "num_envs_per_worker": 1,
        "rollout_fragment_length": 10,  #Per-sampler batch size / Rollout worker batch size 
        "train_batch_size": 50,         #Batch size for training, obtained from the concatenation of rollout worker batches
        "sgd_minibatch_size": 5,  
        "batch_mode": "truncate_episodes",  # or "complete_episodes"
        "timesteps_per_iteration": 0,       #minimal timesteps
        
        # === Resource Settings ===
        "num_gpus": num_gpus,
        "num_cpus_per_worker": 1,
        "num_gpus_per_worker": 0,

        # === Settings for the Trainer process ===
        "gamma": 0.99,
        "lr": 0.0001,
        "model" : {
            "fcnet_hiddens": [4, 4],
            "fcnet_activation": "relu",
            },
        "optimizer": {},
        
        # === Settings for the Env ===
        "horizon" : 30,
        "env": MA_ENV_NAME,
        "env_config": MA_ENV_CONFIG,
            # "observation_space": None,
            # "action_space": None,
        "render_env": False,        #render during training...
        "clip_rewards": None,
        "normalize_actions": True,
                

        # === Multi agent === defines (training?) policies and virtual_agent_id to policy_id mapping.
        "multiagent": multi_agent_config,
        
        # === Exploration settings ===
        "explore": True,
        
        # === Evaluation ===  
        "evaluation_num_workers": 0,
        "evaluation_config": MA_ENV_CONFIG_TEST,
        "evaluation_interval" : 5,
        "evaluation_duration": 1,
        "evaluation_duration_unit": "episodes",
        "custom_eval_function": None,
        
        # === Debug Settings ===
        "log_level": "WARN",
        "callbacks": DefaultCallbacks,
        "simple_optimizer": True,
        "ignore_worker_failures": True,
        "recreate_failed_workers": False,
        "fake_sampler": False,

    }

config = config_concerning_trainer.copy()
config.update(config_concerning_training)


if __name__ == "__main__":
    ###Initialize ray
    print("Ray restarting...")
    ray_init_info = ray.init(local_mode=True, ignore_reinit_error=True)  # in local mode you can debug it 
    print("Dashboard URL: http://{}".format(ray_init_info["webui_url"]))



    ###CHECKPOINT
    print("Defining checkpoint...")
    #Cr??e le dossier checkpoint, cela va save des checkpoints de la policy dans ce dossier. 
    #On pourra par la suite ??valuer (faire des rollout) de cette policy avec l'agent (PPO) sur cet env (CartPole-v1) et pour 200 steps en tappant dans le cmd:
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