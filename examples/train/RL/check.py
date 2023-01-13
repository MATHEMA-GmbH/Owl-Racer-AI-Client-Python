from gym.utils.env_checker import check_env

from owlracer import env

owl_env = env.Env()
owl_env.action_space.seed(42)

observation= owl_env.reset(seed=42)
print(observation)

print(owl_env.observation_space.low)
print(owl_env.observation_space.high)

# actions = all possible step commands

print(f"action space sample: {owl_env.action_space.sample()}")

check_env(owl_env)

#owl_env.close()
