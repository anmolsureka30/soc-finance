import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

# ---- Hyperparameters ----
gamma = 0.99
alpha = 1e-3
epsilon = 1.0
min_epsilon = 0.05
epsilon_decay = 0.9995
buffer_size = 50000
batch_size = 64
target_update_freq = 500
num_episodes = 10000

# ---- Environment ----
env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)
n_states = env.observation_space.n
n_actions = env.action_space.n

# ---- Convert state to one-hot ----
def to_one_hot(state, num_states):
    vec = np.zeros(num_states)
    vec[state] = 1.0
    return vec

# ---- Q-Network ----
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ---- Initialize networks ----
policy_net = DQN(n_states, n_actions)
target_net = DQN(n_states, n_actions)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=alpha)
memory = deque(maxlen=buffer_size)

# ---- Training ----
episode_rewards = []

for episode in range(num_episodes):
    state = env.reset()[0]
    done = False
    total_reward = 0

    while not done:
        state_tensor = torch.FloatTensor(to_one_hot(state, n_states)).unsqueeze(0)
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = policy_net(state_tensor)
                action = torch.argmax(q_values).item()

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        # Sample batch and learn
        if len(memory) >= batch_size:
            batch = random.sample(memory, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.FloatTensor([to_one_hot(s, n_states) for s in states])
            actions = torch.LongTensor(actions).unsqueeze(1)
            rewards = torch.FloatTensor(rewards).unsqueeze(1)
            next_states = torch.FloatTensor([to_one_hot(s, n_states) for s in next_states])
            dones = torch.FloatTensor(dones).unsqueeze(1)

            q_values = policy_net(states).gather(1, actions)
            with torch.no_grad():
                max_next_q = target_net(next_states).max(1)[0].unsqueeze(1)
                target_q = rewards + gamma * max_next_q * (1 - dones)

            loss = nn.MSELoss()(q_values, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Epsilon decay
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    episode_rewards.append(total_reward)

    # Update target network
    if episode % target_update_freq == 0:
        target_net.load_state_dict(policy_net.state_dict())

print("Training finished.")

# ---- Plot rewards ----
plt.figure(figsize=(10, 4))
plt.plot(episode_rewards, alpha=0.3, label="Reward")
if len(episode_rewards) >= 500:
    running_avg = np.convolve(episode_rewards, np.ones(500)/500, mode='valid')
    plt.plot(range(500-1, len(episode_rewards)), running_avg, color='red', label="Running avg (500)")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Episode Reward Over Time (DQN)")
plt.legend()
plt.grid(True)
plt.show()