"""
Reinforcement Learning (DDPG) using Pytroch + singleprocessing.
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
import torch.multiprocessing as mp

import multiprocessing
import os

import test_envs


env = gym.make("bit-flipping-v0")
goal_dim = env.observation_space["desired_goal"].shape[0]
state_dim = env.observation_space["observation"].shape[0] + goal_dim
action_dim = env.action_space.shape[0]

GAMMA = 0.98
NUM_EPOCHS = 1000
MAX_EP = 20
MAX_EP_STEP = 10
BATCH_SIZE = 32


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
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, state, goal):
        x = torch.cat([state, goal], 1)
        x = f.relu(self.linear1(x))
        x = f.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return x


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, state, action, goal):
        x = torch.cat([state, action, goal], 1)
        x = f.relu(self.linear1(x))
        x = f.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class Memory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=int(max_size))
        self.max_size = max_size

    def push_step(self, state, action, reward, next_state, goal, done):
        experience = (state, action, reward, next_state, goal, done)
        self.buffer.append(experience)

    def push_episode_trajectory(self, episode_trajectory):
        for experience in episode_trajectory.memory:
            self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = np.array([])
        action_batch = np.array([])
        reward_batch = np.array([])
        next_state_batch = np.array([])
        goal_batch = np.array([])
        done_batch = np.array([])

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, goal, done = experience
            state_batch = np.append(state_batch, state)
            action_batch = np.append(action_batch, action)
            reward_batch = np.append(reward_batch, reward)
            next_state_batch = np.append(next_state_batch, next_state)
            done_batch = np.append(done_batch, done)
            goal_batch = np.append(goal_batch, goal)

        return state_batch.reshape(batch_size, -1), action_batch.reshape(batch_size, -1), \
               reward_batch.reshape(batch_size, -1), next_state_batch.reshape(batch_size, -1), \
               done_batch.reshape(batch_size, -1), goal_batch.reshape(batch_size, -1)

    def hindsight_experience_replay(self, trajectory, k):
        her_single_episode_trajectory = SingleEpisodeTrajectory()
        for t in range(len(trajectory.memory)):
            for _ in range(k):
                future = np.random.randint(t, len(trajectory.memory))
                goal_ = trajectory.memory[future][3][-goal_dim:]

                state_ = trajectory.memory[t][0]
                action_ = trajectory.memory[t][1]
                next_state_ = trajectory.memory[t][3]
                done = np.array_equal(next_state_[:-goal_dim], goal_)
                reward_ = 0 if done else -1
                her_single_episode_trajectory.add(state_, action_, reward_, next_state_, goal_, done)
        self.push_episode_trajectory(her_single_episode_trajectory)

    def __len__(self):
        return len(self.buffer)


class DDPGAgent:
    def __init__(self, env, hidden_size=256, actor_learning_rate=0.0001, critic_learning_rate=0.0001, gamma=0.98, tau=0.1,
                 max_memory_size=5e5):
        self.goal_dim = goal_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau

        self.action_noise = 1
        self.action_low = env.action_space.low
        self.action_high = env.action_space.high

        # Networks
        self.actor = Actor(self.state_dim + self.goal_dim, hidden_size, self.action_dim)
        self.actor_target = Actor(self.state_dim + self.goal_dim, hidden_size, self.action_dim)
        self.critic = Critic(self.state_dim + self.action_dim + self.goal_dim, hidden_size, 1)
        self.critic_target = Critic(self.state_dim + self.action_dim + self.goal_dim, hidden_size, 1)

        # We initialize the target networks as copies of the original networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        self.memory = Memory(max_memory_size)
        self.critic_criterion = nn.MSELoss()

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

    def get_action(self, state, goal, noise=True):
        state = torch.autograd.Variable(torch.as_tensor(state).float().unsqueeze(0))
        goal = torch.autograd.Variable(torch.as_tensor(goal).float().unsqueeze(0))
        action = self.actor.forward(state, goal)
        action = action.detach().numpy()
        if noise:
            action = np.clip(np.random.normal(action, self.action_noise), self.action_low, self.action_high)
        return np.clip(action, self.action_low, self.action_high)

    def train_networks(self, batch_size):
        state, action, reward, new_state, done, goal = self.memory.sample(batch_size)
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
        optimization_steps = 20
        for _ in range(optimization_steps):
            self.actor_optimizer.zero_grad()
            policy_loss.backward(retain_graph=True)
            self.actor_optimizer.step()

        for _ in range(optimization_steps):
            self.critic_optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            self.critic_optimizer.step()

        # update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        return policy_loss, critic_loss


class Train:
    def __init__(self, agent, env_id="bit-flipping-v0"):
        self.env = gym.make(env_id)
        self.agent = agent

    def run(self):
        for epoch in range(NUM_EPOCHS):
            total_step = 0
            total_episodes = 0
            critic_losses = []
            actor_losses = []
            successes = []
            while total_episodes < MAX_EP:
                obs = self.env.reset()
                episode_reward = 0
                success = False
                single_episode_trajectory = SingleEpisodeTrajectory()
                for t in range(MAX_EP_STEP):
                    state = np.concatenate([obs["observation"], obs["achieved_goal"]])
                    action = self.agent.get_action(state, obs["desired_goal"])
                    next_obs, reward, done, info = self.env.step(action.flatten())
                    success = info["is_success"]
                    next_state = np.concatenate([next_obs["observation"], next_obs["achieved_goal"]])
                    episode_reward += reward
                    single_episode_trajectory.add(state, action, reward, next_state, next_obs["desired_goal"], done)

                    obs = next_obs
                    total_step += 1
                    if done:
                        break
                self.agent.memory.push_episode_trajectory(single_episode_trajectory)
                self.agent.memory.hindsight_experience_replay(single_episode_trajectory, 4)

                if len(self.agent.memory) > BATCH_SIZE:
                    actor_loss, critic_loss = self.agent.train_networks(BATCH_SIZE)
                    actor_losses.append(actor_loss.data.numpy())
                    critic_losses.append(critic_loss.data.numpy())

                # print("Ep: ", total_episodes, " | Ep_r: %.0f" % episode_reward)
                # print("Actor losses: %.0f" % actor_losses, " | Critic losses: %.0f" % critic_losses)
                self.agent.action_noise *= 0.9995
                successes.append(success)
                total_episodes += 1

            print("Training epoch: ", epoch, "successes: ", np.mean(successes))
            print("Training actor losses: ", np.mean(actor_losses), " critic losses: ", np.mean(critic_losses))


if __name__ == '__main__':
    rl_agent = DDPGAgent(env)
    rl_train = Train(rl_agent)
    rl_train.run()
