from collections import OrderedDict
import numpy as np
from gym import GoalEnv, spaces


# environment to evaluate ddpg
class ChaseEnv(GoalEnv):
    def __init__(self, size=2, bound=10, scale=2, reward_type='sparse'):
        # size of the world
        self.size = size

        # type of reward in env
        self.reward_type = reward_type

        # threshold for detecting success
        self.thr = 0.5

        self.observation_space = spaces.Dict({
            'observation': spaces.Box(low=-bound, high=bound, shape=(size,), dtype=np.float32),
            'achieved_goal': spaces.Box(low=-bound, high=bound, shape=(size,), dtype=np.float32),
            'desired_goal': spaces.Box(low=-bound, high=bound, shape=(size,), dtype=np.float32),
        })
        self.obs_space = spaces.Box(low=-bound, high=bound, shape=(size,))

        self.action_space = spaces.Box(low=-bound/scale, high=bound/scale, shape=(size,))

        self.max_steps = size * 20
        self.current_step = 0
        self.state = None
        self.desired_goal = self.obs_space.sample()
        self.reset()

    def _get_obs(self):
        return OrderedDict([
            ('observation', self.state.copy()),
            ('achieved_goal', self.state.copy()),
            ('desired_goal', self.desired_goal.copy())
        ])

    def reset(self):
        # reset goal and state at end of episode
        self.current_step = 0
        self.state = self.obs_space.sample()
        self.desired_goal = self.obs_space.sample()
        return self._get_obs()

    def compute_reward(self, achieved_goal, desired_goal, _info):
        good_done = np.linalg.norm(achieved_goal - desired_goal) <= self.thr
        bad_done = not self.obs_space.contains(achieved_goal)
        is_success = good_done

        if self.reward_type == 'sparse':
            reward = 0 if good_done else -1
        else:
            reward = 5 * self.size if good_done else -10 if bad_done else -np.linalg.norm(achieved_goal - desired_goal) / 200

        return reward, (good_done or bad_done or self.current_step >= self.max_steps), is_success

    def step(self, action):
        # step through the env
        self.state += action
        self.state = np.clip(self.state, self.obs_space.low, self.obs_space.high)
        obs = self._get_obs()
        reward, done, is_success = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], None)
        self.current_step += 1
        info = {'is_success': is_success}
        return obs, reward, done, info

    def render(self):
        # render state of env
        print("\rstate :", np.array_str(self.state),
              "goal :", np.array_str(self.desired_goal))