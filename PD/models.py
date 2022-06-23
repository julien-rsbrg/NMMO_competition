"""
https://github.com/ray-project/ray/blob/releases/1.10.0/rllib/models/torch/torch_modelv2.py
This is for PyTorch but TensorFlow is analogous
"""
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch

class TableOf2_Model(TorchModelV2, torch.nn.Module):
    def __init__(self, observation_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, observation_space, action_space, num_outputs, model_config, name)
        torch.nn.Module.__init__(self)
        
        #Define module attributes and forward() method below.
        self.layer = torch.nn.Linear(1, 2)      
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                      
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        return self.layer(obs), state
