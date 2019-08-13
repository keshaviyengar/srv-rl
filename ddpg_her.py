import random
import time

import numpy as np
import tensorflow as tf
import gym

from collections import OrderedDict

from logger import Logger

STATE_KEYS = ['observation', 'achieved_goal']


# Create replay buffer class for storing experiences
class SingleEpisodeTrajectory:
    def __init__(self):
        self.memory = []

    def add(self, state, action, reward, next_state, done, goal):
        # add tuple of experience in buffer
        self.memory += [(state, action, reward, next_state, done, goal)]

    def clear(self):
        self.memory = []


class DDPGAgent:
    def __init__(self, env, eval_env=None, gamma=0.99, actor_learning_rate=0.01, critic_learning_rate=0.01, tau=1e-3,
                 batch_size=16,
                 buffer_size=5e4, actor_units=256, actor_hidden_layers=1, critic_units=256, critic_hidden_layers=1,
                 k=4, optimization_steps=20, num_epochs=10, num_episodes=20, action_noise=0.5):
        # Assume that the environment is in the format of openai's HER paper
        # (a dictionary of observation, achieved goal and desired goal)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.goal_dim = self.observation_space['achieved_goal'].shape[0]
        self.obs_dim = self.observation_space['observation'].shape[0] + self.goal_dim
        self.action_dim = self.action_space.shape[0]

        # Set limits for action clipping
        self.action_low = env.action_space.low
        self.action_high = env.action_space.high

        # Hyperparameters of RL
        self.gamma = gamma
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.tau = tau
        self.batch_size = batch_size
        self.gradient_norm_clip = None
        self.k = k

        self.optimization_steps = optimization_steps
        self.num_epochs = num_epochs
        self.num_episodes = num_episodes
        self.action_noise = action_noise

        # Initialize experience buffer
        self.memory = []
        self.buffer_size = buffer_size

        # Initialize neural networks
        self._construct_networks(actor_units, actor_hidden_layers, critic_units, critic_hidden_layers)

        self.logger = Logger()
        self.train_env = env
        if eval_env is None:
            self.eval_env = env
        else:
            self.eval_env = eval_env

    def _construct_networks(self, actor_units, actor_hidden_layers, critic_units, critic_hidden_layers):
        # initialize computation graph
        tf.compat.v1.reset_default_graph()
        self.sess = tf.compat.v1.Session()

        # initialize placeholders for computation
        self.R = tf.compat.v1.placeholder(tf.float32, shape=(None, 1), name='reward')
        self.D = tf.compat.v1.placeholder(tf.float32, shape=(None, 1), name='done')
        self.G = tf.compat.v1.placeholder(tf.float32, shape=(None, self.goal_dim), name='goal')
        self.S_0 = tf.compat.v1.placeholder(tf.float32, shape=(None, self.obs_dim), name='state')
        self.S_1 = tf.compat.v1.placeholder(tf.float32, shape=(None, self.obs_dim), name='next_state')

        def _build_a(s, g, units, hidden_layers, scope):
            with tf.compat.v1.variable_scope(scope):
                # state and goal as input for network
                inputs = tf.concat([s, g], 1)
                net = tf.keras.layers.Dense(units, activation='relu', trainable=True)(inputs)
                for _ in range(hidden_layers):
                    net = tf.keras.layers.Dense(units, activation='relu', trainable=True)(net)
                a = tf.keras.layers.Dense(self.action_dim, activation='tanh')(net)
                return a * (self.action_high - self.action_low) / 2 + (self.action_high + self.action_low) / 2

        def _build_c(s, a, g, units, hidden_layers, scope):
            with tf.compat.v1.variable_scope(scope):
                inputs = tf.concat([s, a, g], 1)
                net = tf.keras.layers.Dense(units, activation='relu', trainable=True)(inputs)
                for _ in range(hidden_layers - 1):
                    net = tf.keras.layers.Dense(units, activation='relu', trainable=True)(net)
                return tf.keras.layers.Dense(1)(net)

        # Create actor, critic networks along with target networks
        with tf.compat.v1.variable_scope('Actor'):
            self.a = _build_a(self.S_0, self.G, actor_units, actor_hidden_layers, scope='eval')
            self.a_target = _build_a(self.S_1, self.G, actor_units, actor_hidden_layers, scope='target')

        with tf.compat.v1.variable_scope('Critic'):
            self.q = _build_c(self.S_0, self.a, self.G, critic_units, critic_hidden_layers, scope='eval')
            self.q_target = _build_c(self.S_1, self.a_target, self.G, critic_units, critic_hidden_layers,
                                     scope='target')

        # get list of parameters for each network
        self.a_eval_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='Actor/eval')
        self.a_target_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                                           scope='Actor/target')
        self.c_eval_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                                         scope='Critic/eval')
        self.c_target_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                                           scope='Critic/target')

        # Do a soft update with target networks
        self.soft_update_op = [[tf.compat.v1.assign(target_a, (1 - self.tau) * target_a + self.tau * eval_a),
                                tf.compat.v1.assign(target_c, (1 - self.tau) * target_c + self.tau * eval_c)]
                               for target_a, eval_a, target_c, eval_c in zip(self.a_target_params, self.a_eval_params,
                                                                             self.c_target_params, self.c_eval_params)]
        q_target = self.R + self.gamma * (1 - self.D) * self.q_target

        # loss function for actor and critic networks
        self.c_loss = tf.compat.v1.losses.mean_squared_error(q_target, self.q)
        self.a_loss = - tf.reduce_mean(self.q)

        # Perform optimization based on gradient clipping
        if self.gradient_norm_clip is not None:
            # Initialize critic optimizer
            c_optimizer = tf.compat.v1.train.AdamOptimizer(self.critic_learning_rate)
            c_gradients = c_optimizer.compute_gradients(self.c_loss, var_list=self.c_eval_params)

            # perform gradient clipping
            for i, (grad, var) in enumerate(c_gradients):
                if grad is not None:
                    c_gradients[i] = (tf.clip_by_norm(grad, self.gradient_norm_clip), var)
            self.c_train = c_optimizer.apply_gradients(c_gradients)

            # Initialize actor optimizer
            a_optimizer = tf.compat.v1.train.AdamOptimizer(self.actor_learning_rate)
            a_gradients = a_optimizer.compute_gradients(self.a_loss, var_list=self.a_eval_params)

            # perform gradient clipping
            for i, (grad, var) in enumerate(a_gradients):
                if grad is not None:
                    a_gradients[i] = (tf.clip_by_norm(grad, self.gradient_norm_clip), var)
            self.a_train = c_optimizer.apply_gradients(a_gradients)
        else:
            self.c_train = tf.compat.v1.train.AdamOptimizer(self.critic_learning_rate).minimize(self.c_loss,
                                                                                                var_list=self.c_eval_params)
            self.a_train = tf.compat.v1.train.AdamOptimizer(self.actor_learning_rate).minimize(self.a_loss,
                                                                                               var_list=self.a_eval_params)
        # initialize model saver
        self.saver = tf.compat.v1.train.Saver()

        # variable initialization for tensorflow session
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def choose_action(self, state, goal, action_noise, normal_noise=True):
        action = self.sess.run(self.a, {self.S_0: state, self.G: goal})[0]
        if normal_noise:
            return np.clip(np.random.normal(action, action_noise), self.action_low, self.action_high)
        else:
            return np.clip(action, self.action_low, self.action_high)

    def remember(self, ep_experience):
        self.memory += ep_experience.memory
        if len(self.memory) < self.batch_size:
            return 0, 0

    def train_networks(self, optimization_steps=1):
        # If not enough transitions, do nothing
        if len(self.memory) < self.batch_size:
            return 0, 0

        # Perform optimization for optimization steps
        a_losses = 0
        c_losses = 0
        for _ in range(optimization_steps):
            # Get a mini batch
            mini_batch = np.vstack(random.sample(self.memory, self.batch_size))
            # Stack states, actions and rewards
            ss = np.vstack(mini_batch[:, 0])
            acs = np.vstack(mini_batch[:, 1])
            rs = np.vstack(mini_batch[:, 2])
            nss = np.vstack(mini_batch[:, 3])
            ds = np.vstack(mini_batch[:, 4])
            gs = np.vstack(mini_batch[:, 5])

            # obtain the losses and perform one gradient update step
            a_loss, c_loss, _, _ = self.sess.run([self.a_loss, self.c_loss, self.a_train, self.c_train],
                                                 {self.S_0: ss, self.a: acs, self.R: rs,
                                                  self.S_1: nss, self.D: ds, self.G: gs})
            # accumulate losses over steps
            a_losses += a_loss
            c_losses += c_loss

        return a_losses / optimization_steps, c_losses / optimization_steps

    def update_target_net(self):
        self.sess.run(self.soft_update_op)

    # Util functions
    def convert_dict_to_obs(self, obs_dict, keys):
        """
        :param obs_dict: (dict<np.ndarray>)
        :param keys: Which keys to convert to array
        :return: (np.ndarray)
        """
        # Note: achieved goal is not removed from the observation
        # this is helpful to have a revertible transformation
        if isinstance(self.observation_space, gym.spaces.MultiDiscrete):
            # Special case for multidiscrete
            return np.concatenate([[int(obs_dict[key])] for key in keys])
        return np.concatenate([obs_dict[key] for key in keys])

    def convert_obs_to_dict(self, observations):
        """
        Inverse operation of convert_dict_to_obs
        :param observations: (np.ndarray)
        :return: (OrderedDict<np.ndarray>)
        """
        return OrderedDict([
            ('observation', observations[:self.obs_dim]),
            ('achieved_goal', observations[self.obs_dim:self.obs_dim + self.goal_dim]),
            ('desired_goal', observations[self.obs_dim + self.goal_dim:]),
        ])

    def train(self):
        # initialize buffers for tracking progress
        a_losses = []
        c_losses = []
        ep_mean_r = []
        success_rate = []

        # initialize buffers for episode experience
        single_episode_trajectory = SingleEpisodeTrajectory()
        her_single_episode_trajectory = SingleEpisodeTrajectory()

        # Time performance
        total_step = 0
        start = time.clock()

        # loop for num_epochs
        for i in range(self.num_epochs):
            # track success per epoch
            successes = 0
            ep_total_r = 0

            # loop over episodes
            for n in range(self.num_episodes):
                # reset env
                observation = self.train_env.reset()
                state = self.convert_dict_to_obs(observation, STATE_KEYS)
                goal = self.convert_dict_to_obs(observation, ['desired_goal'])

                # run env for episode length steps
                for ep_step in range(self.train_env.max_steps):
                    # track number of samples
                    total_step += 1

                    # get an action from agent
                    action = self.choose_action([state], [goal], action_noise=self.action_noise)

                    # execute action in env
                    next_observation, reward, done, info = self.train_env.step(action)
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
                    if ep_step == self.train_env.max_steps - 1:
                        replay = 'her'
                        if replay == 'her':
                            for t in range(len(single_episode_trajectory.memory)):
                                # get k future states per timestep
                                for _ in range(self.k):
                                    # get new goal at t_future
                                    future = np.random.randint(t, len(single_episode_trajectory.memory))
                                    goal_ = single_episode_trajectory.memory[future][3][:-self.goal_dim]

                                    state_ = single_episode_trajectory.memory[t][0]
                                    action_ = single_episode_trajectory.memory[t][1]
                                    next_state_ = single_episode_trajectory.memory[t][3]
                                    done = np.array_equal(next_state_[:-self.goal_dim], goal_)
                                    reward_ = 0 if done else -1

                                    # add experience to her buffer
                                    her_single_episode_trajectory.add(state_, action_, reward_, next_state_, done,
                                                                      goal_)
                        # add this her experience to agent buffer
                        self.remember(her_single_episode_trajectory)
                        her_single_episode_trajectory.clear()

                        # add regular experience to agent buffer
                        self.remember(single_episode_trajectory)
                        single_episode_trajectory.clear()

                        self.action_noise *= 0.9995

                        # perform optimization step
                        a_loss, c_loss = self.train_networks(self.optimization_steps)
                        a_losses += [a_loss]
                        c_losses += [c_loss]
                        self.update_target_net()

                    # if episode ends, start new episode
                    if done:
                        break

            # obtain success rate per epoch
            success_rate.append(successes / self.num_episodes)
            ep_mean_r.append(ep_total_r / self.num_episodes)

            # do an evaluation of the current learned policy
            eval_success_rate, eval_ep_mean_r = self.evaluate_model(num_episodes=20)
            self.logger.update_eval_epoch_data(epoch=i, success_rate=eval_success_rate,
                                               mean_reward_per_ep=eval_ep_mean_r)

            # log to epoch data dictionary
            self.logger.update_epoch_data(epoch=i, success_rate=successes / self.num_episodes,
                                          mean_reward_per_ep=ep_total_r / self.num_episodes, actor_losses=a_loss,
                                          critic_losses=c_loss)

            self.logger.print_data('epoch')
            self.logger.print_data('eval')
            print('action_noise: ', self.action_noise)

        # output total training time
        print("training time : %.2f" % (time.clock() - start), "s")
        return eval_success_rate, eval_ep_mean_r

    def evaluate_model(self, num_episodes):
        successes = 0
        ep_total_r = 0
        # Run a number of episodes and evaluate success rate and average reward per episode
        # loop over episodes
        for n in range(num_episodes):
            # reset env
            observation = self.eval_env.reset()
            state = self.convert_dict_to_obs(observation, STATE_KEYS)
            goal = self.convert_dict_to_obs(observation, ['desired_goal'])

            # run env for episode length steps
            for ep_step in range(self.eval_env.max_steps):
                # Get an action from agent
                action = self.choose_action([state], [goal], action_noise=0)

                # Execute action in env
                next_observation, reward, done, info = self.eval_env.step(action)
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

    def save_model(self, name):
        self.saver.save(self.sess, name)

    def restore_model(self, name):
        self.saver.restore(self.sess, name)

from test_envs.bit_flipping_env import BitFlippingEnv
import json
if __name__ == '__main__':
    env = BitFlippingEnv(n_bits=10, continuous=True, max_steps=10)
    #env = gym.make('Distal-1-Tube-Reach-v0')
    # Load json file of parameters
    with open("training_parameters.json", 'r') as f:
        params = json.load(f)

    rl_ = DDPGAgent(env, **params)
    eval_success_rate, eval_ep_mean_reward = rl_.train()
    print('Evaluated final success rate: ,', eval_success_rate)
    print('Evaluated final mean reward per ep,', eval_ep_mean_reward)

    #print('Saving model...')
    #rl_.save_model('models/bit-flipping-env.ckpt')

    # print('Restore model...')
    # rl_2 = DDPGAgent(env, **params)
    # rl_2.restore_model('models/bit-flipping-env.ckpt')

    # print('evaluate model...')
    # success_rate, ep_mean_r = rl_2.evaluate_model(num_episodes=100)
    # print(success_rate)
    # print(ep_mean_r)

