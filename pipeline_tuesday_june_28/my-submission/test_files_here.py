import nmmo
from ijcai2022nmmo import CompetitionConfig
import gym
from rewarding import RewardTeamBasedEnv


from translators import build_translators
from CONFIG import config
config["num_workers"] = 0
config["num_envs_per_worker"] = 1
translators = build_translators(CompetitionConfig())

env = nmmo.Env(CompetitionConfig())
# print('we can clearly set high values for low and high in gym.spaces.Box:')
# print("-------raw observation_space-------")
# print(env.observation_space(agent=1))
# print()

# obs = env.reset()
# # print("-------raw obs-------")
# # print(obs)
# obs_processed = translators[0][0]['obs2obsUsefull'].process(obs[1])
# print(obs_processed)

print("-------test actions-------")
# TeamBasedEnv = RewardTeamBasedEnv(
#     config=CompetitionConfig())
# obs = TeamBasedEnv.reset()

print(nmmo.action.Mage.index)

# action = {
#     nmmo.action.Attack: {
#         nmmo.action.Style: nmmo.action.Mage.index,
#         nmmo.action.Target: 1},
#     nmmo.action.Move: {
#         nmmo.action.Direction: nmmo.action.North.delta}
# }
# actions = {team_id: {agent_id: action for agent_id in range(
#     8)} for team_id in range(16)}
# # print(actions)
# TeamBasedEnv.step(actions)


