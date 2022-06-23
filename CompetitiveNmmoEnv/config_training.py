config_concerning_training = {
        "num_workers" : 2,           # <!> 0 for single-machine training
        "num_envs_per_worker": 1,
        "simple_optimizer": True,
        "ignore_worker_failures": True,
        "batch_mode": "complete_episodes",
        "framework": "torch",
        "disable_env_checking": True,
        "model" : {
            "fcnet_hiddens": [4, 4],
            "fcnet_activation": "relu",
            },
        # "env": "multiagent_PrisonerDilemma",          #other way to specify env. Requires to have register the env before.
        "env": MA_ENV_NAME,
        "env_config": MA_ENV_CONFIG,

        ### MultiAgent config : defines (training?) policies and virtual_agent_id to policy_id mapping.
        "multiagent": multi_agent_config,
        
        ### Evaluation config
        "evaluation_num_workers": 1,
        "evaluation_config": {
            "render_env": True,
            }
    }
