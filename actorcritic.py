import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Hyperparameters
# -------------------------
gamma = 0.99
lr = 1e-3
num_episodes = 1000

# -------------------------
# Environment
# -------------------------
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# -------------------------
# Actor-Critic Network
# -------------------------
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(128, action_dim)
        self.value_head = nn.Linear(128, 1)
        
    def forward(self, x):
        x = self.fc(x)
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value

model = ActorCritic(state_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=lr)

# -------------------------
# Tracking
# -------------------------
all_rewards = []
all_lengths = []
losses = []

# -------------------------
# Training Loop
# -------------------------
for episode in range(num_episodes):
    state = env.reset()[0]
    done = False
    total_reward = 0

    log_probs = []
    values = []
    rewards = []

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        logits, value = model(state_tensor)

        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()

        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        log_prob = dist.log_prob(action)
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)

        state = next_state
        total_reward += reward

    # Compute returns and advantages
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns)
    values = torch.cat(values).squeeze()

    advantages = returns - values.detach()

    # Losses
    actor_loss = -(torch.stack(log_probs) * advantages).mean()
    critic_loss = nn.MSELoss()(values, returns)
    loss = actor_loss + critic_loss

    # Update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Tracking
    all_rewards.append(total_reward)
    all_lengths.append(len(rewards))
    losses.append(loss.item())

    if episode % 50 == 0:
        print(f"Episode {episode}, Reward: {total_reward}, Loss: {loss.item():.3f}")

print("✅ Training complete!")

# -------------------------
# Visualization
# -------------------------

# Reward trend
plt.figure(figsize=(12, 5))
plt.plot(all_rewards, color='blue', alpha=0.3, label='Episode reward')
rolling_mean = np.convolve(all_rewards, np.ones(50)/50, mode='valid')
plt.plot(range(49, len(all_rewards)), rolling_mean, color='red', label='Running avg (50)')
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.title("Actor-Critic: Episode Rewards")
plt.legend()
plt.grid(True)
plt.show()

# Loss trend
plt.figure(figsize=(12, 4))
plt.plot(losses, color='green')
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.title("Actor-Critic: Loss over episodes")
plt.grid(True)
plt.show()

# Episode lengths (optional)
plt.figure(figsize=(12, 4))
plt.plot(all_lengths, color='purple', alpha=0.4, label='Episode length')
rolling_len = np.convolve(all_lengths, np.ones(50)/50, mode='valid')
plt.plot(range(49, len(all_lengths)), rolling_len, color='black', label='Running avg (50)')
plt.xlabel("Episode")
plt.ylabel("Length")
plt.title("Actor-Critic: Episode Lengths")
plt.legend()
plt.grid(True)
plt.show()

# ✅ Done! You now have visuals to analyze learning performance.