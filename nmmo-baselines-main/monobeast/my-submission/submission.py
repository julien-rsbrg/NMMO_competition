from pathlib import Path

import torch
import torch.nn as nn
import tree
from ijcai2022nmmo import Team
from nmmo import config

from torchbeast.monobeast import Net, batch, unbatch
from torchbeast.neural_mmo.train_wrapper import (AttackTeam, FeatureParser,
                                                 TrainWrapper)


class MonobeastBaselineTeam(Team):
    n_action = 5

    def __init__(self,
                 team_id: str,
                 env_config: config.Config,
                 checkpoint_path=None):
        super().__init__(team_id, env_config)
        self.model: nn.Module = Net(None, self.n_action, False)
        if checkpoint_path is not None:
            print(f"load checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            self.model.load_state_dict(checkpoint["model_state_dict"])
        self.feature_parser = FeatureParser()
        self.reset()

    def act(self, observations):
        raw_observations = observations
        observations = self.feature_parser.parse(observations)
        observations = tree.map_structure(
            lambda x: torch.from_numpy(x).view(1, 1, *x.shape), observations)
        obs_batch, ids = batch(observations,
                               self.feature_parser.feature_spec.keys())
        output, _ = self.model(obs_batch)
        output = unbatch(output, ids)
        actions = {i: output[i]["action"].item() for i in output}
        actions = TrainWrapper.transform_action(actions, raw_observations,
                                                self.auxiliary_script)
        return actions

    def reset(self):
        self.auxiliary_script = AttackTeam("auxiliary", self.env_config)


class Submission:
    team_klass = MonobeastBaselineTeam
    init_params = {
        "checkpoint_path":
        Path(__file__).parent / "checkpoints" / "model_113016832.pt"
    }
