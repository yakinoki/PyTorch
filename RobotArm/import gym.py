import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import namedtuple
from itertools import count

# ネットワークの定義
class QNetwork(nn.Module):
    def __init__(self, obs_size, n_actions, n_hidden_channels=50):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_size, n_hidden_channels)
        self.fc2 = nn.Linear(n_hidden_channels, n_hidden_channels)
        self.fc3 = nn.Linear(n_hidden_channels, n_actions)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)

# ハイパーパラメータの設定
gamma = 0.99
alpha = 0.5
num_episodes = 20000

# Experience Replay用のデータ構造
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Replayバッファのクラス
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

# Q関数の初期化
obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n
q_network = QNetwork(obs_size, n_actions)
target_network = QNetwork(obs_size, n_actions)
target_network.load_state_dict(q_network.state_dict())
target_network.eval()

# オプティマイザの設定
optimizer = optim.Adam(q_network.parameters(), lr=1e-2)

# Experience Replay用のバッファを初期化
replay_buffer = ReplayBuffer(capacity=10**6)

# epsilon-greedy explorationの設定
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay = num_episodes // 2
epsilon = epsilon_start

# DQNエージェントの定義
def select_action(state):
    if random.random() < epsilon:
        return torch.tensor([random.randrange(n_actions)])
    else:
        with torch.no_grad():
            return q_network(state).argmax().view(1, 1)

# 学習の設定
BATCH_SIZE = 64
TARGET_UPDATE = 100
episode_rewards = []

# メインの学習ループ
for episode in range(num_episodes):
    state = torch.tensor(env.reset(), dtype=torch.float32)
    done = False
    total_reward = 0

    for t in count():
        action = select_action(state)
        next_state, reward, done, _ = env.step(action.item())
        next_state = torch.tensor(next_state, dtype=torch.float32)
        reward = torch.tensor([reward], dtype=torch.float32)

        replay_buffer.push(state, action, next_state, reward)

        state = next_state
        total_reward += reward.item()

        if len(replay_buffer.memory) >= BATCH_SIZE:
            transitions = replay_buffer.sample(BATCH_SIZE)
            batch = Transition(*zip(*transitions))

            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                    batch.next_state)), dtype=torch.uint8)
            non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            q_values = q_network(state_batch).gather(1, action_batch)

            next_state_values = torch.zeros(BATCH_SIZE)
            next_state_values[non_final_mask] = target_network(non_final_next_states).max(1)[0].detach()

            expected_state_action_values = (next_state_values * gamma) + reward_batch

            loss = nn.MSELoss()(q_values, expected_state_action_values.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            episode_rewards.append(total_reward)
            if episode % 10 == 0:
                print('episode:', episode,
                      'total_reward:', total_reward,
                      'epsilon:', epsilon)
            break

    if episode % TARGET_UPDATE == 0:
        target_network.load_state_dict(q_network.state_dict())

    epsilon -= (epsilon_start - epsilon_end) / epsilon_decay

# エージェントの保存
torch.save(q_network.state_dict(), 'agent.pth')
