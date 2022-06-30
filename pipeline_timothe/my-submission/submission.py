import random
from nmmo.io import action
from ijcai2022nmmo import Team

from env import CompetitionNmmoMultiAgentEnv
from CONFIG import MA_ENV_NAME, config, OurTrainer, CHECKPOINT_NAME, MA_ENV_CONFIG, ConfigTrain
from translators import build_translators
config["num_workers"] = 0
config["num_envs_per_worker"] = 1



checkpoint_path = CHECKPOINT_NAME
# config should rather be the config used for training (not a modified config) preferably for bug
agent_restored = OurTrainer(config, MA_ENV_NAME)
agent_restored.restore(checkpoint_path)

# Define teams


class NotMovingTeam(Team):
    def act(self, observations):
        actions = {}
        for player_idx, obs in observations.items():
            actions[player_idx] = {}
        return actions


class RestoredFromCheckpointTeam(Team):
    # Class parameters:

    # the name of the MA env registered that corresponds to the TeamBased 2 MAenv wrapper.
    MA_ENV_NAME = "CompetitionNmmoMultiAgentEnv"
    # the function that maps the soldier id to the policy id.
    policy_mapping_fn = config["multiagent"]["policy_mapping_fn"]
    translators = build_translators(config=ConfigTrain())
    do_print = False
    agent = agent_restored

    def act(self, observations):
        actions = {}
        for player_idx, observation_soldier in observations.items():
            # Import translators
            obs2obsUsefull = self.translators[0][player_idx]["obs2obsUsefull"]
            actionUsefull2action = self.translators[0][player_idx]["actionUsefull2action"]

            # Nmmo obs -> MA obs -> MA action taken with good policy -> Nmmo action
            observation_usefull = obs2obsUsefull.process(observation_soldier)
            actionUsefull = self.agent.compute_action(
                observation_usefull, policy_id=self.policy_mapping_fn(player_idx))
            actions[player_idx] = actionUsefull2action.traduce(
                observation_soldier, actionUsefull)

            if self.do_print:
                print("Observation soldier:", observation_soldier.keys())
                print("Observation usefull:", observation_usefull)
                print("Action usefull:", actionUsefull)
                print("Action soldier:", actions[player_idx])
                print()

        return actions


class Submission:
    team_klass = RestoredFromCheckpointTeam
    init_params = {}
