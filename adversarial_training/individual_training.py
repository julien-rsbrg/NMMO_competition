print("Imports...")
import gym
import ray
from ray import tune
from ray.rllib.agents import ppo
from ray.tune.logger import pretty_print

import adversarial_training.envs
from adversarial_training.agents import Agent, RandomAgent

import shutil
import os



#Restart ray, generate the Dashboard for performance and parallelization monitoring
#Cliquez sur l'URL qui pop pour allez au Dashboard de ray.
print("Ray restarting...")
ray.shutdown()
info = ray.init(ignore_reinit_error=True)
print("Dashboard URL: http://{}".format(info["webui_url"]))



#Crée le dossier checkpoint, cela va save des checkpoints de la policy dans ce dossier. 
#On pourra par la suite évaluer (faire des rollout) de cette policy avec l'agent (PPO) sur cet env (CartPole-v1) et pour 200 steps en tappant dans le cmd:
# rllib rollout tmp/ppo/CartPole/checkpoint_000001/checkpoint-1 --config "{\"env\": \"CartPole-v1\"}" --run PPO --steps 200
print("Define location for checkpoint...")
CHECKPOINT_ROOT = "tmp/ppo/PD"
shutil.rmtree(CHECKPOINT_ROOT, ignore_errors=True, onerror=None)
ray_results = os.getenv("HOME") + "/ray_results/"
shutil.rmtree(ray_results, ignore_errors=True, onerror=None)



#Define a trainer as a RL algorithm, an env, and a config for training.
print("Define trainer...")
n_players = 4
trainer = ppo.PPOTrainer( env = "PD",
                config = { 
                   "env_config" : { "agents" : [RandomAgent(action_space=gym.spaces.Discrete(2)) for _ in range(n_players)], 
                                    "agent_idx" : 0},       #value is "config" such that an instance of the env would be created using EnvClass(**config) 
                   "framework" : "torch",   #or "tf"             
                   }    
    )



#Training with print evaluation
print("Training...")
N_ITER = 5
s = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f} saved {}"

for n in range(N_ITER):
  result = trainer.train()
#   print(pretty_print(result))
  file_name = trainer.save(CHECKPOINT_ROOT)
  print(s.format(
    n + 1,
    result["episode_reward_min"],
    result["episode_reward_mean"],
    result["episode_reward_max"],
    result["episode_len_mean"],
    file_name
   ))
print("End of training.")



#Check the model
print("Policy and model:")
policy = trainer.get_policy()
model = policy.model
# print(model.base_model.summary())   #Ca c'est pour TF, je sais pas quoi mettre pour torch





print("Done.")