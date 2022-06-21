print("Imports...")
import shutil
import os

import ray
from ray import tune

from CartPole.env import *
from CartPole.CONFIG import *



print("Configurating...")
#Import trainer and config for trainer (algo hyperparameters, some training parameters)
from CartPole.trainer import SimpleDQNTrainer, SimplePPOTrainer, config_concerning_trainer
#Import config for training (env/env_config, framework, ressource allocation and other non-algorithm dependant training parameters)
from CartPole.config_training import config_concerning_training
#Merge config
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
# rllib rollout tmp/ppo/CartPole/checkpoint_000006/checkpoint-6 --config "{\"env\": \"CartPole6V1\"}" --run PPO --steps 200
print("Define location for checkpoint...")
CHECKPOINT_ROOT = "tmp/ppo/CartPole"
shutil.rmtree(CHECKPOINT_ROOT, ignore_errors=True, onerror=None)
ray_results = os.getenv("HOME") + "/ray_results/"
shutil.rmtree(ray_results, ignore_errors=True, onerror=None)



###TRAINING
print("Start training...")
if RUN_WITH_TUNE:   #j'ai pas testé avec tune
    # Tune is the system for keeping track of all of the running jobs, originally for
    # hyperparameter tuning, not sure it works
    tune.registry.register_trainable("SimplePPOTrainer", SimplePPOTrainer)
    tune.registry.register_trainable("DQNTrainer", SimpleDQNTrainer)
    stop = {
            "training_iteration": NUM_ITERATIONS  # Each iteration is some number of episodes
        }
    results = tune.run("SimplePPOTrainer", stop=stop, config=config, verbose=1, checkpoint_freq=10)

else:
    trainer = SimplePPOTrainer(config, env="multiagent_CartPole-v1")
    
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