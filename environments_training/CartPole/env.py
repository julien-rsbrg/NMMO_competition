from ray.rllib.env.multi_agent_env import MultiAgentEnv, make_multi_agent
from ray.tune.registry import register_env
import gym

from CartPole.utils import *

from typing import *


#No need to implement an Env, since CartPole is already register as a individual gym env. We will here register the
#MA env as a gym (MA) env.
cart_pole_env_creator = lambda config : gym.make("CartPole-v1")
def env_creator(env_config):
    num_agents = env_config["num_agents"]
    CartPoleEnv_MA = make_multi_agent(cart_pole_env_creator)
    return CartPoleEnv_MA({"num_agents": num_agents})
register_env("multiagent_CartPole-v1", env_creator)