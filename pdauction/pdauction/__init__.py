from gym.envs.registration import register

register(
    id="pdauction/Auctioneer-v0",
    entry_point="pdauction.envs:AuctioneerEnv",
)