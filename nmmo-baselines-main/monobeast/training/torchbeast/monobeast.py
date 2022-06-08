# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import pprint
import threading
import time
import timeit
import traceback
import typing
from typing import Dict, List

os.environ["OMP_NUM_THREADS"] = "1"  # Necessary for multithreading.

from pathlib import Path

import torch
from ijcai2022nmmo import CompetitionConfig, TeamBasedEnv
from torch import multiprocessing as mp
from torch import nn
from torch.nn import functional as F

from torchbeast.core import file_writer, prof, vtrace
from torchbeast.neural_mmo.monobeast_wrapper import \
    MonobeastWrapper as Environment
from torchbeast.neural_mmo.net import NMMONet
from torchbeast.neural_mmo.train_wrapper import TrainWrapper

to_torch_dtype = {
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}

# yapf: disable
parser = argparse.ArgumentParser(description="PyTorch Scalable Agent")

parser.add_argument("--env", type=str, default="IJCAI2022-NMMO",
                    help="Gym environment.")
parser.add_argument("--mode", default="train",
                    choices=["train", "test", "test_render"],
                    help="Training or test mode.")
parser.add_argument("--xpid", default=None,
                    help="Experiment id (default: None).")

# Training settings.
parser.add_argument("--disable_checkpoint", action="store_true",
                    help="Disable saving checkpoint.")
parser.add_argument("--savedir", default="~/logs/torchbeast",
                    help="Root dir where experiment data will be saved.")
parser.add_argument("--num_actors", default=4, type=int, metavar="N",
                    help="Number of actors (default: 4).")
parser.add_argument("--total_steps", default=100000, type=int, metavar="T",
                    help="Total environment steps to train for.")
parser.add_argument("--batch_size", default=8, type=int, metavar="B",
                    help="Learner batch size.")
parser.add_argument("--unroll_length", default=80, type=int, metavar="T",
                    help="The unroll length (time dimension).")
parser.add_argument("--num_buffers", default=None, type=int,
                    metavar="N", help="Number of shared-memory buffers.")
parser.add_argument("--num_learner_threads", "--num_threads", default=1, type=int,
                    metavar="N", help="Number learner threads.")
parser.add_argument("--disable_cuda", action="store_true",
                    help="Disable CUDA.")
parser.add_argument("--use_lstm", action="store_true",
                    help="Use LSTM in agent model.")
parser.add_argument("--checkpoint_interval", default=600, type=int, metavar="T",
                    help="checkpoint interval (default: 10min).")
# Loss settings.
parser.add_argument("--entropy_cost", default=0.0006,
                    type=float, help="Entropy cost/multiplier.")
parser.add_argument("--baseline_cost", default=0.5,
                    type=float, help="Baseline cost/multiplier.")
parser.add_argument("--discounting", default=0.99,
                    type=float, help="Discounting factor.")
parser.add_argument("--reward_clipping", default="none",
                    choices=["abs_one", "none"],
                    help="Reward clipping.")

# Optimizer settings.
parser.add_argument("--learning_rate", default=0.00048,
                    type=float, metavar="LR", help="Learning rate.")
parser.add_argument("--alpha", default=0.99, type=float,
                    help="RMSProp smoothing constant.")
parser.add_argument("--momentum", default=0, type=float,
                    help="RMSProp momentum.")
parser.add_argument("--epsilon", default=0.01, type=float,
                    help="RMSProp epsilon.")
parser.add_argument("--grad_norm_clipping", default=40.0, type=float,
                    help="Global gradient norm clip.")
# yapf: enable

logging.basicConfig(
    format=("[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] "
            "%(message)s"),
    level=logging.INFO,
)

Buffers = typing.Dict[str, typing.List[torch.Tensor]]


def compute_baseline_loss(advantages, mask=None):
    if mask is not None:
        mask = torch.ones_like(advantages)
    loss = (advantages**2)
    loss *= mask
    return torch.sum(loss) / torch.sum(mask)


def compute_entropy_loss(logits, mask=None):
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    if mask is None:
        mask = torch.ones_like(logits)
    else:
        mask = mask.unsqueeze(dim=-1).expand_as(logits)
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    loss = policy * log_policy
    loss *= mask
    return torch.sum(loss) / torch.sum(mask)


def compute_policy_gradient_loss(logits, actions, advantages, mask=None):
    if mask is None:
        mask = torch.ones_like(advantages)
    cross_entropy = F.nll_loss(
        F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
        target=torch.flatten(actions, 0, 1),
        reduction="none",
    )
    cross_entropy = cross_entropy.view_as(advantages)
    loss = cross_entropy * advantages.detach()
    loss *= mask
    return torch.sum(loss) / torch.sum(mask)


def store(env_output, agent_output, agent_state, buffers: Buffers,
          initial_agent_state_buffers, free_indices, t):
    # return
    indices_iter = iter(free_indices)
    """Store tensor in buffer."""
    for agent_id in env_output.keys():
        index = next(indices_iter)

        for key, val in env_output[agent_id].items():
            buffers[key][index][t, ...] = val
        for key, val in agent_output[agent_id].items():
            buffers[key][index][t, ...] = val
        for i, tensor in enumerate(agent_state):
            initial_agent_state_buffers[index][i][...] = tensor


def batch(env_output: Dict, filter_keys: List[str]):
    """Transform agent-wise env_output to bach format."""
    filter_keys = list(filter_keys)
    obs_batch = {key: [] for key in filter_keys}
    agent_ids = []
    for agent_id, out in env_output.items():
        agent_ids.append(agent_id)
        for key, val in out.items():
            if key in filter_keys:
                obs_batch[key].append(val)
    for key, val in obs_batch.items():
        obs_batch[key] = torch.cat(val, dim=1)

    return obs_batch, agent_ids


def unbatch(agent_output: Dict, agent_ids):
    """Transform agent_output to agent-wise format."""
    unbatched_agent_output = {key: {} for key in agent_ids}
    for key, val in agent_output.items():
        for i, agent_id in enumerate(agent_ids):
            unbatched_agent_output[agent_id][
                key] = val[:, i]  # val shape: [1, B, ...]
    return unbatched_agent_output


def act(
    flags,
    actor_index: int,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    model: torch.nn.Module,
    buffers: Buffers,
    initial_agent_state_buffers,
):
    try:
        logging.info("Actor %i started.", actor_index)
        timings = prof.Timings()  # Keep track of how fast things are.

        gym_env = create_env(flags)
        seed = actor_index ^ int.from_bytes(os.urandom(4), byteorder="little")
        gym_env.seed(seed)
        env = Environment(gym_env)
        env_output = env.initial()
        agent_state = model.initial_state(batch_size=1)
        env_output_batch, agent_ids = batch(
            env_output, filter_keys=gym_env.observation_space.keys())
        agent_output_batch, unused_state = model(env_output_batch, agent_state)
        agent_output = unbatch(agent_output_batch, agent_ids)
        while True:

            free_indices = [free_queue.get() for _ in range(flags.num_agents)]
            if None in free_indices:
                break

            # Write old rollout end.
            store(env_output, agent_output, agent_state, buffers,
                  initial_agent_state_buffers, free_indices, 0)

            # Do new rollout.
            for t in range(flags.unroll_length):
                timings.reset()

                with torch.no_grad():
                    # batch
                    env_output_batch, agent_ids = batch(
                        env_output,
                        filter_keys=gym_env.observation_space.keys())
                    # forward inference
                    agent_output_batch, agent_state = model(
                        env_output_batch, agent_state)
                    # unbatch
                    agent_output = unbatch(agent_output_batch, agent_ids)
                    # extract actions
                    actions = {
                        agent_id: agent_output[agent_id]["action"].item()
                        for agent_id in agent_output
                    }

                timings.time("model")

                env_output = env.step(actions)

                timings.time("step")

                store(env_output, agent_output, (), buffers,
                      initial_agent_state_buffers, free_indices, t + 1)

                timings.time("write")
            [full_queue.put(index) for index in free_indices]

        if actor_index == 0:
            logging.info("Actor %i: %s", actor_index, timings.summary())

    except KeyboardInterrupt:
        pass  # Return silently.
    except Exception as e:
        logging.error("Exception in worker process %i", actor_index)
        traceback.print_exc()
        print()
        raise e


def get_batch(
        flags,
        free_queue: mp.SimpleQueue,
        full_queue: mp.SimpleQueue,
        buffers: Buffers,
        initial_agent_state_buffers,
        timings,
        lock=threading.Lock(),
):
    with lock:
        timings.time("lock")
        indices = [full_queue.get() for _ in range(flags.batch_size)]
        timings.time("dequeue")
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1)
        for key in buffers
    }
    initial_agent_state = tuple()
    if flags.use_lstm:
        initial_agent_state = (torch.cat(ts, dim=1) for ts in zip(
            *[initial_agent_state_buffers[m] for m in indices]))
    timings.time("batch")
    for m in indices:
        free_queue.put(m)
    timings.time("enqueue")
    batch = {
        k: t.to(device=flags.device, non_blocking=True)
        for k, t in batch.items()
    }
    if flags.use_lstm:
        initial_agent_state = tuple(
            t.to(device=flags.device, non_blocking=True)
            for t in initial_agent_state)
    timings.time("device")
    return batch, initial_agent_state


def learn(
        flags,
        actor_model,
        model,
        batch,
        initial_agent_state,
        optimizer,
        scheduler,
        lock=threading.Lock(),  # noqa: B008
):
    """Performs a learning (optimization) step."""
    with lock:
        learner_outputs, unused_state = model(batch,
                                              initial_agent_state,
                                              training=True)

        # Take final value function slice for bootstrapping.
        bootstrap_value = learner_outputs["baseline"][-1]

        # Move from obs[t] -> action[t] to action[t] -> obs[t].
        batch = {key: tensor[1:] for key, tensor in batch.items()}
        learner_outputs = {
            key: tensor[:-1]
            for key, tensor in learner_outputs.items()
        }

        rewards = batch["reward"]
        if flags.reward_clipping == "abs_one":
            clipped_rewards = torch.clamp(rewards, -1, 1)
        elif flags.reward_clipping == "none":
            clipped_rewards = rewards

        discounts = (~batch["done"]).float() * flags.discounting
        mask = (~batch["done"]).float()  # mask dead agent

        vtrace_returns = vtrace.from_logits(
            behavior_policy_logits=batch["policy_logits"],
            target_policy_logits=learner_outputs["policy_logits"],
            actions=batch["action"],
            discounts=discounts,
            rewards=clipped_rewards,
            values=learner_outputs["baseline"],
            bootstrap_value=bootstrap_value,
        )

        pg_loss = compute_policy_gradient_loss(
            learner_outputs["policy_logits"],
            batch["action"],
            vtrace_returns.pg_advantages,
            mask=mask,
        )
        baseline_loss = flags.baseline_cost * compute_baseline_loss(
            vtrace_returns.vs - learner_outputs["baseline"], mask=mask)
        entropy_loss = flags.entropy_cost * compute_entropy_loss(
            learner_outputs["policy_logits"], mask=mask)

        total_loss = pg_loss + baseline_loss + entropy_loss

        episode_returns = batch["episode_return"][batch["done"]]
        episode_steps = batch["episode_step"][batch["done"]]

        optimizer.zero_grad()
        total_loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(),
                                             flags.grad_norm_clipping)
        optimizer.step()
        scheduler.step()
        actor_model.load_state_dict(model.state_dict())
        stats = {
            "episode_returns": tuple(episode_returns.cpu().numpy()),
            "mean_episode_return": torch.mean(episode_returns).item(),
            "mean_episode_step": torch.mean(episode_steps.float()).item(),
            "total_loss": total_loss.item(),
            "pg_loss": pg_loss.item(),
            "baseline_loss": baseline_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "advantage": torch.mean(vtrace_returns.pg_advantages).item(),
            "rho": torch.exp(vtrace_returns.log_rhos).mean().item(),
            "grad_norm": grad_norm.item(),
        }

        return stats


def create_buffers(flags, observation_space, num_actions) -> Buffers:
    T = flags.unroll_length
    # observation_space is a dict
    obs_specs = {
        key: dict(size=(T + 1, *val.shape),
                  dtype=to_torch_dtype[val.dtype.name])
        for key, val in observation_space.items()
    }
    specs = dict(
        reward=dict(size=(T + 1, ), dtype=torch.float32),
        done=dict(size=(T + 1, ), dtype=torch.bool),
        episode_return=dict(size=(T + 1, ), dtype=torch.float32),
        episode_step=dict(size=(T + 1, ), dtype=torch.int32),
        policy_logits=dict(size=(T + 1, num_actions), dtype=torch.float32),
        baseline=dict(size=(T + 1, ), dtype=torch.float32),
        last_action=dict(size=(T + 1, ), dtype=torch.int64),
        action=dict(size=(T + 1, ), dtype=torch.int64),
    )
    specs.update(obs_specs)
    buffers: Buffers = {key: [] for key in specs}
    for _ in range(flags.num_buffers):
        for key in buffers:
            buffers[key].append(torch.zeros(**specs[key]).share_memory_())
    return buffers


def start_process(flags, ctx, model, actor_processes, free_queue, full_queue,
                  buffers, initial_agent_state_buffers):
    """Periodically restart actor process to prevent OOM, which may be caused by pytorch share_memory"""
    if len(actor_processes) > 0:
        logging.critical("Stoping actor process...")
        for actor in actor_processes:
            actor.terminate()
            actor.join()
            actor.close()

    while not free_queue.empty():
        free_queue.get()
    while not full_queue.empty():
        full_queue.get()
    for m in range(flags.num_buffers):
        free_queue.put(m)

    logging.critical("Starting actor process...")
    actor_processes = []
    for i in range(flags.num_actors):
        actor = ctx.Process(
            target=act,
            args=(
                flags,
                i,
                free_queue,
                full_queue,
                model,
                buffers,
                initial_agent_state_buffers,
            ),
        )
        actor.start()
        actor_processes.append(actor)
        time.sleep(0.5)
    return actor_processes


def train(flags):  # pylint: disable=too-many-branches, too-many-statements
    if flags.xpid is None:
        flags.xpid = "torchbeast-%s" % time.strftime("%Y%m%d-%H%M%S")
    plogger = file_writer.FileWriter(xpid=flags.xpid,
                                     xp_args=flags.__dict__,
                                     rootdir=flags.savedir)
    checkpointpath = Path(flags.savedir).joinpath(flags.xpid)
    if flags.num_buffers is None:  # Set sensible default for num_buffers.
        flags.num_buffers = max(2 * flags.num_agents * flags.num_actors,
                                flags.batch_size)
    if flags.num_actors >= flags.num_buffers:
        raise ValueError("num_buffers should be larger than num_actors")
    if flags.num_buffers < flags.batch_size:
        raise ValueError("num_buffers should be larger than batch_size")

    T = flags.unroll_length
    B = flags.batch_size

    flags.device = None
    if not flags.disable_cuda and torch.cuda.is_available():
        logging.info("Using CUDA.")
        flags.device = torch.device("cuda")
    else:
        logging.info("Not using CUDA.")
        flags.device = torch.device("cpu")

    env = create_env(flags)

    model = Net(env.observation_space, env.action_space.n, flags.use_lstm)
    buffers = create_buffers(flags, env.observation_space, env.action_space.n)

    model.share_memory()

    # Add initial RNN state.
    initial_agent_state_buffers = []
    for _ in range(flags.num_buffers):
        state = model.initial_state(batch_size=1)
        for t in state:
            t.share_memory_()
        initial_agent_state_buffers.append(state)

    ctx = mp.get_context("fork")
    free_queue = ctx.SimpleQueue()
    full_queue = ctx.SimpleQueue()

    actor_processes = start_process(flags, ctx, model, [], free_queue,
                                    full_queue, buffers,
                                    initial_agent_state_buffers)

    learner_model = Net(env.observation_space.shape, env.action_space.n,
                        flags.use_lstm).to(device=flags.device)

    optimizer = torch.optim.RMSprop(
        learner_model.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha,
    )

    def lr_lambda(epoch):
        return 1 - min(epoch * T * B, flags.total_steps) / flags.total_steps

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    logger = logging.getLogger("logfile")
    stat_keys = [
        "total_loss",
        "mean_episode_return",
        "mean_episode_step",
        "pg_loss",
        "baseline_loss",
        "entropy_loss",
        "rho",
        "advantage",
        "grad_norm",
    ]
    logger.info("# Step\t%s", "\t".join(stat_keys))

    step, stats = 0, {}

    def batch_and_learn(i, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal step, stats
        timings = prof.Timings()
        while step < flags.total_steps:
            timings.reset()
            batch, agent_state = get_batch(
                flags,
                free_queue,
                full_queue,
                buffers,
                initial_agent_state_buffers,
                timings,
            )
            stats = learn(flags, model, learner_model, batch, agent_state,
                          optimizer, scheduler)
            timings.time("learn")
            with lock:
                to_log = dict(step=step)
                to_log.update({k: stats[k] for k in stat_keys})
                plogger.log(to_log)
                step += T * B

        if i == 0:
            logging.info("Batch and learn: %s", timings.summary())

    threads = []
    for i in range(flags.num_learner_threads):
        thread = threading.Thread(target=batch_and_learn,
                                  name="batch-and-learn-%d" % i,
                                  args=(i, ))
        thread.start()
        threads.append(thread)

    def checkpoint():
        if flags.disable_checkpoint:
            return
        logging.info("Saving checkpoint to %s", checkpointpath)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "flags": vars(flags),
            },
            checkpointpath.joinpath(f"model_{step}.pt"),
        )

    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer()
        while step < flags.total_steps:
            start_step = step
            start_time = timer()
            time.sleep(5)

            if timer() - last_checkpoint_time > flags.checkpoint_interval:
                checkpoint()
                last_checkpoint_time = timer()
                actor_processes = start_process(flags, ctx, model,
                                                actor_processes, free_queue,
                                                full_queue, buffers,
                                                initial_agent_state_buffers)

            sps = (step - start_step) / (timer() - start_time)
            if stats.get("episode_returns", None):
                mean_return = ("Return per episode: %.1f. " %
                               stats["mean_episode_return"])
            else:
                mean_return = ""
            total_loss = stats.get("total_loss", float("inf"))
            logging.info(
                "Steps %i @ %.1f SPS. Loss %f. %sStats:\n%s",
                step,
                sps,
                total_loss,
                mean_return,
                pprint.pformat(stats),
            )
    except KeyboardInterrupt:
        return  # Try joining actors then quit.
    else:
        for thread in threads:
            thread.join()
        logging.info("Learning after %d steps.", step)
    finally:
        for _ in range(flags.num_actors * flags.num_agents):
            free_queue.put(None)
        for actor in actor_processes:
            actor.join(timeout=1)

    checkpoint()
    plogger.close()


Net = NMMONet


def create_env(flags):
    cfg = CompetitionConfig()
    cfg.NMAPS = 400
    return TrainWrapper(TeamBasedEnv(config=cfg))


def main(flags):
    if flags.mode == "train":
        train(flags)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    flags = parser.parse_args()
    flags.num_agents = int(CompetitionConfig.NENT / CompetitionConfig.NPOP)
    main(flags)
