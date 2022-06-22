import gym

from contrib.actors import Actor, RandomActor
from contrib.env_wrappers import IndividualEnvWrapper
from CompetitiveEnv.env import CompetitionMultiAgentEnv

#My parameters
n_teams = 2
n_soldiers = 2
do_render = True
only_one_episode = True

#Define env
soldier_action_space = gym.spaces.Discrete(n_soldiers * n_teams)
MA_env = CompetitionMultiAgentEnv(n_teams, n_soldiers)
env = IndividualEnvWrapper(     MA_env = MA_env, 
                                actors={agent_id : RandomActor(action_space=soldier_action_space) for agent_id in MA_env._agents_ids},
                                agent_id="0_0",
                                )

#Test indiv env with action being permanently 1               
if __name__ == "__main__":
    observation = env.reset(seed=42)
    done = False
    for t in range(1000):
        if do_render: env.render()
        action = 1
        observation, reward, done, info = env.step(action)
        print("Timestep: ", t)
        print("Actions: ", action)
        print("Next Observation: ", observation)
        print("Reward: ", reward)
        print("\n\n")
        if done:
            if do_render: env.render()
            if only_one_episode: break
            observation = env.reset()

    env.close()
    print("Done")