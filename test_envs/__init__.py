from gym.envs.registration import register

register(
    id="bit-flipping-v0",
    entry_point='test_envs.envs:BitFlippingEnv',
    kwargs={"n_bits": 10, "continuous": True, "max_steps": 10, "discrete_obs_space": False}
)
