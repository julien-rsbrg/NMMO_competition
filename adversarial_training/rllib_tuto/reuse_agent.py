import ray
from ray.rllib.agents.ppo import PPOTrainer
import gym

ray.init()


agent = PPOTrainer(     env = "CartPole-v1",
                        config = { 
                                "env_config" : {},       
                                "framework" : "torch",               
                                } )

agent.restore(checkpoint_path="tmp/ppo/CartPole/checkpoint_000001/checkpoint-1")


env = gym.make("CartPole-v1")
obs = env.reset()
for i in range(100):
    # env.render()
    action = agent.compute_action(obs)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()