"""
Reinforcement Learning (A3C) using Pytroch + multiprocessing.
The most simple implementation for continuous action.
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

import test_envs


UPDATE_GLOBAL_ITER = 5
GAMMA = 0.99 # Discount factor
MAX_EP = 100
MAX_EP_STEP = 10

env = gym.make("bit-flipping-v0")
state_dim = env.observation_space["observation"].shape[0]
goal_dim = env.observation_space["desired_goal"].shape[0]
action_dim = env.action_space.shape[0]
N_S = state_dim + action_dim
N_A = action_dim


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, state):
        x = f.relu(self.linear1(state))
        x = f.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return x


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = f.relu(self.linear1(x))
        x = f.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class Memory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=int(max_size))
        self.max_size = max_size

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3):
        super(SharedAdam, self).__init__(params, lr=lr)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


class DDPGAgent:
    def __init__(self, env, hidden_size=256, actor_learning_rate=1e-4, critic_learning_rate=1e-4, gamma=0.98, tau=1e-2,
                 max_memory_size=5e5, global_agent=False):
        self.obs_dim = state_dim
        self.goal_dim = goal_dim
        self.state_dim = self.obs_dim + self.goal_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau

        self.action_noise = 1
        self.action_low = env.action_space.low
        self.action_high = env.action_space.high

        # Networks
        self.actor = Actor(self.state_dim, hidden_size, self.action_dim)
        self.actor_target = Actor(self.state_dim, hidden_size, self.action_dim)
        self.critic = Critic(self.state_dim + self.action_dim, hidden_size, 1)
        self.critic_target = Critic(self.state_dim + self.action_dim, hidden_size, 1)
        if global_agent:
            self.actor.share_memory()
            self.actor_target.share_memory()
            self.critic.share_memory()
            self.critic_target.share_memory()

        # We initialize the target networks as copies of the original networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        self.memory = Memory(max_memory_size)
        self.critic_criterion = nn.MSELoss()
        if global_agent:
            self.actor_optimizer = SharedAdam(self.actor.parameters(), lr=actor_learning_rate)
            self.critic_optimizer = SharedAdam(self.critic.parameters(), lr=critic_learning_rate)
        else:
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

    def get_action(self, state):
        state = torch.autograd.Variable(torch.as_tensor(state).float())
        action = self.actor.forward(state)
        action = action.detach().numpy()
        np.clip(np.random.normal(action, self.action_noise), self.action_low, self.action_high)
        return action

    def train_networks(self, batch_size):
        state, action, reward, new_state, done = self.memory.sample(batch_size)
        states = torch.FloatTensor(state)
        actions = torch.FloatTensor(action)
        rewards = torch.FloatTensor(reward)
        new_states = torch.FloatTensor(new_state)
        dones = torch.FloatTensor(done)

        # Critic Loss
        q_vals = self.critic.forward(states, actions)
        next_actions = self.actor.forward(new_states)
        next_q = self.critic_target.forward(new_states, next_actions.detach())
        q_prime = rewards + self.gamma * next_q
        critic_loss = self.critic_criterion(q_vals, q_prime)

        # Actor Loss
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()

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


class Worker(mp.Process):
    def __init__(self, global_episode, global_ep_reward, result_queue, name, env_id="bit-flipping-v0"):
        super(Worker, self).__init__()
        self.name = 'worker_%i' % name
        self.env = gym.make(env_id)
        self.agent = DDPGAgent(self.env)
        self.global_ep, self.global_ep_r, self.res_queue = global_episode, global_ep_reward, result_queue

    def run(self):
        total_step = 1
        while self.global_ep.value < MAX_EP:
            obs = self.env.reset()
            episode_reward = 0
            for t in range(MAX_EP_STEP):
                state = np.concatenate([obs["observation"], obs["achieved_goal"]])
                if self.name == 'worker_0':
                    self.env.render()
                action = self.agent.get_action(state)
                next_obs, reward, done, info = self.env.step(action)
                print("done: ", done)
                next_state = np.concatenate([next_obs["observation"], next_obs["achieved_goal"]])
                episode_reward += reward
                self.agent.memory.push(state, action, reward, next_state, done)

                if len(self.agent.memory) > self.agent.memory.max_size:
                    self.agent.train_networks(self.agent.memory.max_size)
                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global agent and assign to local net
                    print("Do some syncing here...")
                    if done:
                        self.record(episode_reward)

                obs = next_obs
                total_step += 1
            self.agent.action_noise *= 0.999
        self.res_queue.put(None)

    def record(self, ep_r):
        with self.global_ep.get_lock():
            self.global_ep.value += 1
        with self.global_ep_r.get_lock():
            if self.global_ep_r.value == 0:
                self.global_ep_r.value = ep_r
            else:
                self.global_ep_r.value = self.global_ep_r.value * self.agent.gamma + ep_r * (1 - self.agent.gamma)
        self.res_queue.put(self.global_ep_r.value)
        print(
           self.name,
            "Ep:", self.global_ep.value,
            "| Ep_r: %.0f" % self.global_ep_r.value,
        )


if __name__ == '__main__':
    global_agent = DDPGAgent(env=env, global_agent=True)
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
    workers = []
    for i in range(1):
        workers.append(Worker(global_ep, global_ep_r, res_queue, i))
    for worker in workers:
        worker.start()
    res = []
    while True:
        print("getting reward from queue")
        r = res_queue.get()
        print("received reward from queue")
        if r is not None:
            res.append(r)
        else:
            break
    for worker in workers:
        worker.join()

    import matplotlib.pyplot as plt
    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()