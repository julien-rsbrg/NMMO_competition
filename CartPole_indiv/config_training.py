from ray.rllib.policy.policy import PolicySpec

import CartPole_indiv.CONFIG as CONFIG


from typing import Dict 

# see https://github.com/ray-project/ray/blob/releases/1.10.0/rllib/agents/trainer.py
config_concerning_training = {
        "num_workers" : 0,           # <!> 0 for single-machine training
        "simple_optimizer": True,
        "ignore_worker_failures": True,
        "batch_mode": "complete_episodes",
        "framework": "torch",

        "env": "CartPole-v1",
        "env_config": {},

    }
