from gym.envs.registration import register

register(
    id='powertac_wm-v0',
    entry_point='gym_powertac.envs:PowerTAC_WM',
)
