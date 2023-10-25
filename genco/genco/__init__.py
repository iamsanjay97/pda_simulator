from gym.envs.registration import register

register(
    id="genco/CPGenCo-v0",
    entry_point="genco.envs:CPGenCoEnv",
)