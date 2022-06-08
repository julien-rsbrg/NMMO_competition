import torch
import torch.nn as nn
import torch.nn.functional as F

from torchbeast.core.mask import MaskedPolicy


class NMMONet(nn.Module):

    def __init__(self, observation_space, num_actions, use_lstm=False):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=17,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=16,
                      kernel_size=3,
                      stride=1,
                      padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=16,
                      out_channels=16,
                      kernel_size=3,
                      stride=1,
                      padding=1), nn.ReLU())
        self.core = nn.Linear(16 * 15 * 15, 512)
        self.policy = nn.Linear(512, num_actions)
        self.baseline = nn.Linear(512, 1)

    def initial_state(self, batch_size=1):
        return tuple()

    def forward(self, input_dict, state=(), training=False):
        # [T, B, ...]
        assert "va" in input_dict
        terrain, camp, entity = input_dict["terrain"], input_dict[
            "camp"], input_dict["entity"]
        terrain = F.one_hot(terrain, num_classes=6).permute(0, 1, 4, 2, 3)
        camp = F.one_hot(camp, num_classes=4).permute(0, 1, 4, 2, 3)

        # print(terrain.shape, camp.shape, entity.shape)
        x = torch.cat([terrain, camp, entity], dim=2)

        T, B, C, H, W = x.shape
        # assert C == 17 and H == W == 15
        x = torch.flatten(x, 0, 1)
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.core(x))

        logits = self.policy(x)
        baseline = self.baseline(x)

        va = input_dict.get("va", None)
        if va is not None:
            va = torch.flatten(va, 0, 1)

        dist = MaskedPolicy(logits, valid_actions=va)
        if not training:
            action = dist.sample()
            action = action.view(T, B)
        else:
            action = None
        policy_logits = dist.logits.view(T, B, -1)
        baseline = baseline.view(T, B)
        output = dict(policy_logits=policy_logits, baseline=baseline)
        if action is not None:
            output["action"] = action
        return (output, tuple())