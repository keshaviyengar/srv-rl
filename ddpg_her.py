"""
Reinforcement Learning (DDPG) using Pytorch + single environment implementation
First try out out continous bit-flipping env, then distal-1 then distal-2
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import gym
from collections import deque
import random

import test_envs
import ctm2_envs


# Create replay buffer class for storing experiences
class SingleEpisodeTrajectory:
    def __init__(self):
        self.memory = []

    def add(self, state, action, reward, next_state, goal, done):
        # add tuple of experience in buffer
        self.memory += [(state, action, reward, next_state, goal, done)]

    def clear(self):
        self.memory = []


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_layers, output_size):
        super(Actor, self).__init__()
        self.input = nn.Linear(input_size, hidden_size)
        self.hidden = nn.ModuleList()
        for _ in range(hidden_layers):
            self.hidden.append(nn.Linear(hidden_size, hidden_size))
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, state, goal):
        x = torch.cat([state, goal], 1)
        x = f.relu(self.input(x))
        for layer in self.hidden:
            x = f.relu(layer(x))
        x = torch.tanh(self.output(x))
        return x


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_layers, output_size):
        super(Critic, self).__init__()
        self.input = nn.Linear(input_size, hidden_size)
        self.hidden = nn.ModuleList()
        for _ in range(hidden_layers):
            self.hidden.append(nn.Linear(hidden_size, hidden_size))
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, state, action, goal):
        x = torch.cat([state, action, goal], 1)
        x = f.relu(self.input(x))
        for layer in self.hidden:
            x = f.relu(layer(x))
        x = self.output(x)
        return x


class Memory:
    def __init__(self, batch_size, max_size, k, goal_dim):
        self.max_size = max_size
        self.buffer = deque(maxlen=int(max_size))
        self.max_size = max_size
        self.batch_size = batch_size
        self.k = k
        self.goal_dim = goal_dim

    def push_step(self, state, action, reward, next_state, goal, done):
        experience = (state, action, reward, next_state, goal, done)
        self.buffer.append(experience)

    def push_episode_trajectory(self, episode_trajectory):
        for experience in episode_trajectory.memory:
            self.buffer.append(experience)

    def sample(self):
        state_batch = np.array([])
        action_batch = np.array([])
        reward_batch = np.array([])
        next_state_batch = np.array([])
        goal_batch = np.array([])
        done_batch = np.array([])

        batch = random.sample(self.buffer, self.batch_size)

        for experience in batch:
            state, action, reward, next_state, goal, done = experience
            state_batch = np.append(state_batch, state)
            action_batch = np.append(action_batch, action)
            reward_batch = np.append(reward_batch, reward)
            next_state_batch = np.append(next_state_batch, next_state)
            done_batch = np.append(done_batch, done)
            goal_batch = np.append(goal_batch, goal)

        return state_batch.reshape(self.batch_size, -1), action_batch.reshape(self.batch_size, -1), \
               reward_batch.reshape(self.batch_size, -1), next_state_batch.reshape(self.batch_size, -1), \
               done_batch.reshape(self.batch_size, -1), goal_batch.reshape(self.batch_size, -1)

    def hindsight_experience_replay(self, trajectory):
        her_single_episode_trajectory = SingleEpisodeTrajectory()
        for t in range(len(trajectory.memory)):
            for _ in range(self.k):
                future = np.random.randint(t, len(trajectory.memory))
                goal_ = trajectory.memory[future][3][-self.goal_dim:]

                state_ = trajectory.memory[t][0]
                action_ = trajectory.memory[t][1]
                next_state_ = trajectory.memory[t][3]
                done = np.array_equal(next_state_[:-self.goal_dim], goal_)
                reward_ = 0 if done else -1
                her_single_episode_trajectory.add(state_, action_, reward_, next_state_, goal_, done)
        self.push_episode_trajectory(her_single_episode_trajectory)

    def __len__(self):
        return len(self.buffer)


class DDPGAgent:
    def __init__(self, env, num_epochs=1000, num_episodes=20, actor_mlp_hidden_layers=1, actor_mlp_units=256,
                 critic_mlp_hidden_layers=1, critic_mlp_units=256, actor_learning_rate=0.0001,
                 critic_learning_rate=0.0001, gamma=0.98, tau=0.1, action_noise=1,
                 batch_size=32, buffer_size=5e5, future_k=4):
        self.env = env
        self.eval_env = env

        self.goal_dim = env.observation_space["desired_goal"].shape[0]
        self.state_dim = env.observation_space["observation"].shape[0] + self.goal_dim
        self.action_dim = env.action_space.shape[0]
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        self.num_epochs = num_epochs
        self.num_episodes = num_episodes
        self.max_steps = self.env.max_steps

        self.k = future_k

        self.action_noise = action_noise
        self.action_low = env.action_space.low
        self.action_high = env.action_space.high

        # Networks
        self.actor = Actor(input_size=self.state_dim + self.goal_dim, hidden_size= actor_mlp_units, hidden_layers=actor_mlp_hidden_layers, output_size=self.action_dim)
        self.actor_target = Actor(input_size=self.state_dim + self.goal_dim, hidden_size= actor_mlp_units, hidden_layers=actor_mlp_hidden_layers, output_size=self.action_dim)
        self.critic = Critic(input_size=self.state_dim + self.action_dim + self.goal_dim, hidden_size=critic_mlp_units, hidden_layers=critic_mlp_hidden_layers, output_size=1)
        self.critic_target = Critic(input_size=self.state_dim + + self.action_dim + self.goal_dim, hidden_size=critic_mlp_units, hidden_layers=critic_mlp_hidden_layers, output_size=1)

        # We initialize the target networks as copies of the original networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        self.memory = Memory(self.batch_size, self.buffer_size, self.k, self.goal_dim)
        self.critic_criterion = nn.MSELoss()

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

    def get_action(self, state, goal, noise=True):
        state = torch.autograd.Variable(torch.as_tensor(state).float().unsqueeze(0))
        goal = torch.autograd.Variable(torch.as_tensor(goal).float().unsqueeze(0))
        if noise:
            action = self.actor.forward(state, goal)
        else:
            action = self.actor_target(state, goal)
        action = action.detach().numpy()
        if noise:
            action = np.clip(np.random.normal(action, self.action_noise), self.action_low, self.action_high)
        return np.clip(action, self.action_low, self.action_high)

    def train_networks(self):
        state, action, reward, new_state, done, goal = self.memory.sample()
        states = torch.FloatTensor(state)
        actions = torch.FloatTensor(action)
        rewards = torch.FloatTensor(reward)
        new_states = torch.FloatTensor(new_state)
        dones = torch.FloatTensor(done)
        goals = torch.FloatTensor(goal)

        # Critic Loss
        q_vals = self.critic.forward(states, actions, goals)
        next_actions = self.actor.forward(new_states, goals)
        next_q = self.critic_target.forward(new_states, next_actions.detach(), goals)
        q_prime = rewards + self.gamma * (torch.ones(dones.size()) - dones) * next_q
        critic_loss = self.critic_criterion(q_vals, q_prime)

        # Actor Loss
        policy_loss = -self.critic.forward(states, self.actor.forward(states, goals), goals).mean()

        # update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        return policy_loss, critic_loss

    def train(self):
        total_step = 1

        for epoch in range(self.num_epochs):
            total_episodes = 0
            critic_losses = []
            actor_losses = []
            successes = []
            errors = []
            while total_episodes < self.num_episodes:
                obs = self.env.reset()
                episode_reward = 0
                success = False
                error = 0
                single_episode_trajectory = SingleEpisodeTrajectory()
                for t in range(self.max_steps):
                    state = np.concatenate([obs["observation"], obs["achieved_goal"]])
                    action = self.get_action(state, obs["desired_goal"])
                    next_obs, reward, done, info = self.env.step(action.flatten())
                    success = info["is_success"]
                    # error = info["error"]
                    next_state = np.concatenate([next_obs["observation"], next_obs["achieved_goal"]])
                    episode_reward += reward
                    single_episode_trajectory.add(state, action, reward, next_state, next_obs["desired_goal"], done)

                    obs = next_obs
                    total_step += 1
                    if done:
                        break
                self.memory.push_episode_trajectory(single_episode_trajectory)
                self.memory.hindsight_experience_replay(single_episode_trajectory)
                if len(self.memory) > self.batch_size:
                    actor_loss, critic_loss = self.train_networks()
                    actor_losses.append(actor_loss.data.numpy())
                    critic_losses.append(critic_loss.data.numpy())
                # print("Ep: ", total_episodes, " | Ep_r: %.0f" % episode_reward)
                # print("Actor losses: %.0f" % actor_losses, " | Critic losses: %.0f" % critic_losses)
                self.action_noise *= 0.9995
                successes.append(success)
                # errors.append(error)
                total_episodes += 1

            print("Training epoch: ", epoch, "successes: ", np.mean(successes))
            print("Training actor losses: ", np.mean(actor_losses), " critic losses: ", np.mean(critic_losses))
            # print("Training mean error: ", np.mean(errors))
            self.evaluate()

    def evaluate(self):
        total_step = 1
        successes = []
        mean_ep_rewards = []
        errors = []

        total_episodes = 0
        while total_episodes < self.num_episodes:
            obs = self.eval_env.reset()
            episode_reward = 0
            success = False
            error = 0
            for t in range(self.max_steps):
                state = np.concatenate([obs["observation"], obs["achieved_goal"]])
                action = self.get_action(state, obs["desired_goal"], noise=False)
                next_obs, reward, done, info = self.eval_env.step(action.flatten())
                success = info["is_success"]
                # error = info["error"]
                next_state = np.concatenate([next_obs["observation"], next_obs["achieved_goal"]])
                episode_reward += reward
                obs = next_obs
                total_step += 1
                if done:
                    break
            total_episodes += 1
            successes.append(success)
            # errors.append(error)
            mean_ep_rewards.append(episode_reward)
        print("eval successes: ", np.mean(successes), "eval mean reward: ", np.mean(mean_ep_rewards))
