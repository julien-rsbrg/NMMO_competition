"""
See https://github.com/ray-project/ray/blob/releases/1.0.0/rllib/policy/torch_policy.py
This is for PyTorch but TensorFlow is analogous
"""
from ray.rllib import TorchPolicy

from PD.models import MyModel


class PrisonerPolicy(TorchPolicy):

    def __init__(self, observation_space, action_space, config):
        model = MyModel(observation_space, 
                                action_space,    
                                num_outputs=1,
                                model_config=config,
                                name='MyModel')
        self.action_space = action_space
        self.observation_space = observation_space
        super().__init__(observation_space=observation_space,
                         action_space=action_space,
                         model=model,
                         config=config)