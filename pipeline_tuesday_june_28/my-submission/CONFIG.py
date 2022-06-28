import os
import sys
from typing import Dict
import gym.spaces
from ray.rllib.agents.ppo import ppo
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.policy.policy import PolicySpec

from ijcai2022nmmo import CompetitionConfig

n_teams = 16
n_soldiers = 8
n_agents = n_teams * n_soldiers


class ConfigTrain(CompetitionConfig):
    RENDER = False


class ConfigTest(CompetitionConfig):
    RENDER = True


MA_ENV_NAME = "CompetitionNmmoMultiAgentEnv"
MA_ENV_CONFIG = {"Config_class": ConfigTrain}
MA_ENV_CONFIG_TEST = {"Config_class": ConfigTest}

### ======================== WHAT TO CHANGE IS BELOW ======================== ###


# ENV CONFIG
# obs space of the individual agent ie the soldier
soldier_observation_space = gym.spaces.Dict({
    "perso_info": gym.spaces.Box(low=-40, high=500, shape=(3,)),
    "other_entities_info": gym.spaces.Box(low=-500, high=500, shape=(4, 7)),
    "N_ennemy_players": gym.spaces.Box(low=-1, high=n_soldiers*3, shape=(1,)),
    "N_NPCs": gym.spaces.Box(low=-1, high=n_soldiers*3, shape=(1,)),
    "N_friends": gym.spaces.Box(low=-1, high=n_soldiers*3, shape=(1,)),
})

# action space of the individual agent ie the soldier
soldier_action_space = gym.spaces.Discrete(9)

# TRAINING CONFIG


class OurTrainer(ppo.PPOTrainer):
    pass


config_concerning_trainer = ppo.DEFAULT_CONFIG.copy()

RUN_WITH_TUNE = False                                   # no tuning for now
# number of training iterations .train()
NUM_ITERATIONS = 100
# path to save checkpoints, such that checkpoint folder is .../pipeline_X/starter_kit/my_submission/+CHECKPOINT_ROOT
CHECKPOINT_ROOT = "tmp/ppo/NMMO_MA"

# EVALUATION CONFIG
# name always of the form "checkpoint_?/checkpoint-?"   such that the checkpoint is at .../my_submission/checkpoint_?/checkpoint-?
CHECKPOINT_NAME = "checkpoint_000003/checkpoint-3"

### MULTI AGENT CONFIG ###


# maps agent_id to policy_id
def policy_map_fn(agent_id: str, _episode=None, _worker=None, **_kwargs) -> str:
    """
    Maps agent_id to policy_id
    """
    return "soldier_policy"


def get_multiagent_policies() -> Dict[str, PolicySpec]:
    policies: Dict[str, PolicySpec] = {}  # policy_id to policy_spec

    # Policy of the VirtualAgent "prisoner"
    policies['soldier_policy'] = PolicySpec(
        policy_class=None,  # use default in trainer, or could be YourHighLevelPolicy
        observation_space=soldier_observation_space,
        action_space=soldier_action_space,
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
config_concerning_training = {
    # === General settings ===
    "framework": "tf2",
    # "disable_env_checking": True,

    # === Worker & sampling settings ===
    # <!> 0 for single-machine training. Number of rollout worker actors to create for parallel sampling
    "num_workers": 0,
    "num_envs_per_worker": 1,
    "rollout_fragment_length": 10,  # Per-sampler batch size / Rollout worker batch size, will adjust itself so that : num_workers * num_envs_per_worker * rollout_fragment_length = k * train_batch_size
    # Batch size for training, obtained from the concatenation of rollout worker batches
    "train_batch_size": 50,
    "sgd_minibatch_size": 25,
    "batch_mode": "truncate_episodes",  # "truncate_episodes" or "complete_episodes"
    "timesteps_per_iteration": 0,  # minimal timesteps

    # === Resource Settings ===
    "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
    "num_cpus_per_worker": 2,
    "num_gpus_per_worker": 0,

    # === Settings for the Trainer process ===
    "gamma": 0.99,
    "lr": 0.0001,
    "model": {
        "fcnet_hiddens": [32, 32],
        "fcnet_activation": "relu",
    },
    "optimizer": {},

    # === Settings for the Env ===
    "horizon": 30,
    "env": MA_ENV_NAME,
    "env_config": MA_ENV_CONFIG,
    "observation_space": soldier_observation_space,
    "action_space": soldier_action_space,
    "render_env": False,  # render during training...
    "clip_rewards": None,
    "normalize_actions": True,


    # === Multi agent === defines (training?) policies and virtual_agent_id to policy_id mapping.
    "multiagent": multi_agent_config,

    # === Exploration settings ===
    "explore": True,

    # === Evaluation ===
    "evaluation_num_workers": 0,
    "evaluation_config": MA_ENV_CONFIG_TEST,
    "evaluation_interval": 5,
    "evaluation_duration": 1,
    "evaluation_duration_unit": "episodes",
    "custom_eval_function": None,

    # === Debug Settings ===
    "log_level": "WARN",
    "callbacks": DefaultCallbacks,
    "simple_optimizer": True,
    "ignore_worker_failures": True,
    # "recreate_failed_workers": False,
    "fake_sampler": False,

}

config = config_concerning_trainer.copy()
config.update(config_concerning_training)


### ================================================================= ###
