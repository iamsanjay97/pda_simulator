from gym.envs.registration import register

register(
    id="genco/EnergyBroker-v0",
    entry_point="broker.envs:BrokerEnv",
)