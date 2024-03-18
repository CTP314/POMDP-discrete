from ml_collections import ConfigDict
from typing import Tuple
from gym.envs.registration import register
from configs.envs.terminal_fns import finite_horizon_terminal

env_name_fn = lambda l: f"S5-{l}-v0"

def create_fn(config: ConfigDict) -> Tuple[ConfigDict, str]:
    length = config.env_name
    env_name = env_name_fn(length)
    register(
        env_name,
        entry_point="envs.regular:S5POMDP",
        kwargs=dict(
            length=length,
        ),
        max_episode_steps=length,  # NOTE: has to define it here
    )

    del config.create_fn
    return config, env_name

def get_config():
    config = ConfigDict()
    config.create_fn = create_fn

    config.env_type = "regular_s5_fixed"
    config.terminal_fn = finite_horizon_terminal

    config.eval_interval = 500
    config.save_interval = 500
    config.eval_episodes = 100

    # [1, 2, 5, 10, 30, 50, 100, 300, 500, 1000]
    config.env_name = 10

    return config