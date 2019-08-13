import time
from collections import OrderedDict
from gym import spaces
import numpy as np

from ddpg_her import DDPGAgent
from ddpg_her import SingleEpisodeTrajectory
from logger import Logger

import json

STATE_KEYS = ['observation', 'achieved_goal']


class RLController:
    def __init__(self, env, agent_kwargs):
        self.env = env
        self.agent = DDPGAgent(env, **agent_kwargs)
        self.logger = Logger()

    def train(self, replay='her', k=4, optimization_steps=20, num_epochs=100, num_episodes=50,
              episode_length=20, action_noise=0.5):
        # Does learning and evaluation together
        # initialize buffers for tracking progress
        a_losses = []
        c_losses = []
        ep_mean_r = []
        success_rate = []

        # initialize buffers for episode experience
        single_episode_trajectory = SingleEpisodeTrajectory()
        her_single_episode_trajectory = SingleEpisodeTrajectory()

        # Time performance of network
        total_step = 0
        start = time.clock()

        # loop for num_epochs
        for i in range(num_epochs):
            # track success per epoch
            successes = 0
            ep_total_r = 0

            # loop over episodes
            for n in range(num_episodes):
                # reset env
                observation = self.env.reset()
                state = self.convert_dict_to_obs(observation, STATE_KEYS)
                goal = self.convert_dict_to_obs(observation, ['desired_goal'])

                # run env for episode length steps
                for ep_step in range(episode_length):
                    # track number of samples
                    total_step += 1

                    # get an action from agent
                    action = self.agent.choose_action([state], [goal], action_noise=action_noise)

                    # execute action in env
                    next_observation, reward, done, info = self.env.step(action)
                    next_state = self.convert_dict_to_obs(next_observation, STATE_KEYS)

                    # track reward and add to replay buffer
                    ep_total_r += reward
                    # keep track of success
                    successes += info['is_success']
                    single_episode_trajectory.add(state, action, reward, next_state, done, goal)
                    state = next_state

                    # record data for episode
                    self.logger.update_episode_data(epoch=i, episode=n, step=ep_step,
                                                    state_joint=observation['observation'],
                                                    state_ag=observation['achieved_goal'], action=action,
                                                    next_state_joint=next_observation['observation'],
                                                    next_state_ag=next_observation['achieved_goal'],
                                                    reward=reward, success=info['is_success'])
                    if ep_step == episode_length - 1:
                        if replay == 'her':
                            for t in range(len(single_episode_trajectory.memory)):
                                # get k future states per timestep
                                for _ in range(k):
                                    # get new goal at t_future
                                    future = np.random.randint(t, len(single_episode_trajectory.memory))
                                    goal_ = single_episode_trajectory.memory[future][3][:-self.agent.goal_dim]

                                    state_ = single_episode_trajectory.memory[t][0]
                                    action_ = single_episode_trajectory.memory[t][1]
                                    next_state_ = single_episode_trajectory.memory[t][3]
                                    done = np.array_equal(next_state_[:-self.agent.goal_dim], goal_)
                                    reward_ = 0 if done else -1

                                    # add experience to her buffer
                                    her_single_episode_trajectory.add(state_, action_, reward_, next_state_, done, goal_)

                        # add this her experience to agent buffer
                        self.agent.remember(her_single_episode_trajectory)
                        her_single_episode_trajectory.clear()

                        # add regular experience to agent buffer
                        self.agent.remember(single_episode_trajectory)
                        single_episode_trajectory.clear()

                        action_noise *= 0.9995

                        # perform optimization step
                        a_loss, c_loss = self.agent.train_networks(optimization_steps)
                        a_losses += [a_loss]
                        c_losses += [c_loss]
                        self.agent.update_target_net()

                    # if episode ends, start new episode
                    if done:
                        break


            # obtain success rate per epoch
            success_rate.append(successes / num_episodes)
            ep_mean_r.append(ep_total_r / num_episodes)

            # do an evaluation of the current learned policy
            eval_success_rate, eval_ep_mean_r = self.evaluate_controller(eval_env=self.env, num_episodes=20,
                                                                         episode_length=episode_length)
            self.logger.update_eval_epoch_data(epoch=i, success_rate=eval_success_rate,
                                               mean_reward_per_ep=eval_ep_mean_r)

            # log to epoch data dictionary
            self.logger.update_epoch_data(epoch=i, success_rate=successes / num_episodes,
                                          mean_reward_per_ep=ep_total_r / num_episodes, actor_losses=a_loss,
                                          critic_losses=c_loss)

            self.logger.print_data('epoch')
            self.logger.print_data('eval')
            print('action_noise: ', action_noise)

        eval_success_rate, eval_ep_mean_r = self.evaluate_controller(eval_env=self.env, num_episodes=20,
                                                                     episode_length=episode_length)

        # output total training time
        print("training time : %.2f" % (time.clock() - start), "s")
        return eval_success_rate, eval_ep_mean_r

    def evaluate_controller(self, eval_env=None, num_episodes=50, episode_length=10):
        successes = 0
        ep_total_r = 0
        # Run a number of episodes and evaluate success rate and average reward per episode
        # loop over episodes
        for n in range(num_episodes):
            # reset env
            observation = eval_env.reset()
            state = self.convert_dict_to_obs(observation, STATE_KEYS)
            goal = self.convert_dict_to_obs(observation, ['desired_goal'])

            # run env for episode length steps
            for ep_step in range(episode_length):
                # Get an action from agent
                action = self.agent.choose_action([state], [goal], action_noise=0)

                # Execute action in env
                next_observation, reward, done, info = self.env.step(action)
                next_state = self.convert_dict_to_obs(next_observation, STATE_KEYS)

                # Track reward and add to replay buffer
                ep_total_r += reward
                successes += info['is_success']
                state = next_state

                # if episode ends, start new episode
                if done:
                    break

            # obtain success rate per epoch
        success_rate = successes / num_episodes
        ep_mean_r = ep_total_r / num_episodes

        return success_rate, ep_mean_r

    # Util functions
    def convert_dict_to_obs(self, obs_dict, keys):
        """
        :param obs_dict: (dict<np.ndarray>)
        :param keys: Which keys to convert to array
        :return: (np.ndarray)
        """
        # Note: achieved goal is not removed from the observation
        # this is helpful to have a revertible transformation
        if isinstance(self.agent.observation_space, spaces.MultiDiscrete):
            # Special case for multi-discrete
            return np.concatenate([[int(obs_dict[key])] for key in keys])
        return np.concatenate([obs_dict[key] for key in keys])

    def convert_obs_to_dict(self, observations):
        """
        Inverse operation of convert_dict_to_obs
        :param observations: (np.ndarray)
        :return: (OrderedDict<np.ndarray>)
        """
        return OrderedDict([
            ('observation', observations[:self.agent.obs_dim]),
            ('achieved_goal', observations[self.agent.obs_dim:self.agent.obs_dim + self.agent.goal_dim]),
            ('desired_goal', observations[self.agent.obs_dim + self.agent.goal_dim:]),
        ])


from test_envs.bit_flipping_env import BitFlippingEnv

if __name__ == '__main__':
    env = BitFlippingEnv(n_bits=10, continuous=True, max_steps=10)
    #env = gym.make('Distal-1-Tube-Reach-v0')
    # Load json file of parameters
    with open("ddpg_her_parameters.json", 'r') as f:
        params = json.load(f)
        agent_kwargs = params['agent_hyperparams']
        training_kwargs = params['training_hyperparams']

    rl_controller = RLController(env, agent_kwargs)
    rl_controller.train(**training_kwargs)
