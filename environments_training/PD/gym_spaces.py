"""
You pick a chicken to befriend and then get the location of that chicken

A couple of articles on hierarchical reinforcement learning
https://thegradient.pub/the-promise-of-hierarchical-reinforcement-learning/
https://towardsdatascience.com/hierarchical-reinforcement-learning-a2cca9b76097

"""
import numpy as np

from gym import spaces

prisoner_action_space = spaces.Discrete(2)
prisoner_observation_space = spaces.Discrete(1)