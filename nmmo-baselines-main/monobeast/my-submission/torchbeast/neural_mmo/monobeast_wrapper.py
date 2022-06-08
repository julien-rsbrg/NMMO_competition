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
"""The environment class for MonoBeast."""
from typing import Dict

import torch
import tree
from numpy import ndarray
from torchbeast.neural_mmo.train_wrapper import TrainWrapper


class MonobeastWrapper:
    def __init__(self, env: TrainWrapper):
        if not isinstance(env, TrainWrapper):
            raise RuntimeError(f"gym_env is not instance of TrainWrapper.")
        self.gym_env = env

    def initial(self):
        initial_frame = self.gym_env.reset()
        output = self.format_output(initial_frame)
        self.episode_return = {agent_id: 0 for agent_id in self.gym_env.agents}
        self.episode_step = {agent_id: 0 for agent_id in self.gym_env.agents}
        return output

    def step(self, actions: Dict[int, int]):
        frame, reward, done, unused_info = self.gym_env.step(actions)
        for agent_id, d in done.items():
            if not d:
                self.episode_step[agent_id] += 1
                self.episode_return[agent_id] += reward[agent_id]
        episode_step = self.episode_step
        episode_return = self.episode_return
        if all(done.values()):
            frame = self.gym_env.reset()
            self.episode_step = {k: 0 for k in self.episode_step}
            self.episode_return = {k: 0 for k in self.episode_return}
        output = self.format_output(frame, actions, reward, done,
                                    episode_return, episode_step)
        return output

    def close(self):
        self.gym_env.close()

    def format_output(self,
                      frame,
                      actions=None,
                      reward=None,
                      done=None,
                      episode_return=None,
                      episode_step=None):
        assert isinstance(frame,
                          dict), f"only support frame in the format of dict"
        if actions is None:
            actions = {agent_id: 0 for agent_id in self.gym_env.agents}
        if reward is None:
            reward = {agent_id: 0 for agent_id in self.gym_env.agents}
        if done is None:
            done = {agent_id: 1 for agent_id in self.gym_env.agents}
        if episode_return is None:
            episode_return = {agent_id: 0 for agent_id in self.gym_env.agents}
        if episode_step is None:
            episode_step = {agent_id: 0 for agent_id in self.gym_env.agents}

        def _format(x):
            if isinstance(x, ndarray):
                return torch.from_numpy(x).view(1, 1, *x.shape)
            elif isinstance(x, (int, float)):
                return torch.tensor(x).view(1, 1)
            else:
                raise RuntimeError

        frame = tree.map_structure(_format, frame)
        actions = tree.map_structure(_format, actions)
        reward = tree.map_structure(_format, reward)
        done = tree.map_structure(_format, done)
        episode_return = tree.map_structure(_format, episode_return)
        episode_step = tree.map_structure(_format, episode_step)

        output = {}
        for agent_id in self.gym_env.agents:
            o = {}
            o.update(frame[agent_id])
            o["last_action"] = actions[agent_id]
            o["reward"] = reward[agent_id]
            o["done"] = done[agent_id]
            o["episode_return"] = episode_return[agent_id]
            o["episode_step"] = episode_step[agent_id]
            output[agent_id] = o

        return output
