"""
File for training agents on CompetitiveEnv as multi-agents problem.
"""

print("Imports...")
import argparse
from gym.spaces import Dict, Discrete, Tuple, MultiDiscrete
import logging
import os

import ray
from ray import tune
from ray.tune import register_env
import ray.rllib as rllib
from ray.rllib.examples.env.two_step_game import TwoStepGame
from ray.rllib.env.multi_agent_env import ENV_STATE
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.agents.qmix import *

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run", type=str, default="PG", help="The RLlib-registered algorithm to use."
)
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "tfe", "torch"],
    default="torch",
    help="The DL framework specifier.",
)
parser.add_argument("--num-cpus", type=int, default=1)
parser.add_argument(
    "--mixer",
    type=str,
    default="qmix",
    choices=["qmix", "vdn", "none"],
    help="The mixer model to use.",
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--stop-iters", type=int, default=200, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=70000, help="Number of timesteps to train."
)
parser.add_argument(
    "--stop-reward", type=float, default=1000000000000, help="Reward at which we stop training."
)
parser.add_argument(
    "--local-mode",
    action="store_true",
    help="Init Ray in local mode for easier debugging.",
)

if __name__ == "__main__":
    args = parser.parse_args()

    print("Ray init...")
    ray.init(num_cpus=args.num_cpus or None, local_mode=args.local_mode)






    ### GROUPING ENV
    print("Configurating...")
    ###############################################################################################
    from CompetitiveEnv.env import CompetitionMultiAgentEnv
    MA_ENV_CLASS = CompetitionMultiAgentEnv
    MA_ENV_NAME = "CompetitionMultiAgentEnv"
    MAG_ENV_NAME = "GroupedCompetitionMultiAgentEnv"
    n_teams = 2
    n_soldiers = 2
    n_agents = n_teams * n_soldiers
    ENV_CONFIG = {
            "n_teams": n_teams,
            "n_soldiers": n_soldiers,
        }
    ###############################################################################################
    
    
    obs_space = Tuple(
        [
            Dict(
                {
                    "obs": MultiDiscrete([2, 2, 2, 3]),
                    ENV_STATE: MultiDiscrete([2, 2, 2]),
                }
            ),
            Dict(
                {
                    "obs": MultiDiscrete([2, 2, 2, 3]),
                    ENV_STATE: MultiDiscrete([2, 2, 2]),
                }
            ),
        ]
    )
    act_space = Tuple(
        [
            TwoStepGame.action_space,
            TwoStepGame.action_space,
        ]
    )

    # grouping = {"team" + str(num_team) : [str(num_team) + "_" + str(num_soldier) for num_soldier in range(n_soldiers)] for num_team in range(n_teams)}
    # print("Grouping:", grouping)
    # register_env(
    #     MAG_ENV_NAME,
    #     lambda config: MA_ENV_CLASS(n_teams = config["n_teams"], n_soldiers = config["n_soldiers"]).with_agent_groups(
    #         grouping, obs_space=None, act_space=None
    #     ),
    # )

        
    config = {
        "env": MA_ENV_NAME,
        "env_config": ENV_CONFIG,
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "framework": args.framework,
        "disable_env_checking": True,
    }

    stop = {
        "episode_reward_mean": args.stop_reward,
        "timesteps_total": args.stop_timesteps,
        "training_iteration": args.stop_iters,
    }

    print("Tuning...")
    results = tune.run(args.run, stop=stop, config=config, verbose=2)

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)

    ray.shutdown()