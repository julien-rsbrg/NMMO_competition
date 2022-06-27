"""
File for training agents on the Prisoner's Dilemma MA env.
"""

print("Imports...")
from typing import *
import shutil
import os

import ray
from ray.rllib.agents.ppo import ppo



### CONFIGURATION ###
print("Importing environment and configuration from CONFIG...")
import env  # this register the MA env
from CONFIG import MA_ENV_NAME, config, ConfigTest, CHECKPOINT_ROOT, NUM_ITERATIONS



if __name__ == "__main__":
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