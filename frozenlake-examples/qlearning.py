import numpy as np
import gym
import matplotlib.pyplot as plt

# Setup environment
env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)
n_states = env.observation_space.n
n_actions = env.action_space.n

# Hyperparameters
gamma = 0.99
alpha = 0.7
epsilon = 1.0
min_epsilon = 0.05
epsilon_decay = 0.9999  # Slower decay to ensure more exploration
num_episodes = 300_000  # More episodes to guarantee enough learning

Q = np.zeros((n_states, n_actions))
rewards = []

# Training loop
for episode in range(num_episodes):
    state = env.reset()[0]
    done = False
    total_reward = 0

    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
        state = next_state
        total_reward += reward

    # Decay epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    rewards.append(total_reward)

print("Training finished.")

# Value function and policy
V = np.max(Q, axis=1)
policy = np.argmax(Q, axis=1)

V_grid = V.reshape(8, 8)
policy_grid = policy.reshape(8, 8)

print("Final state value:")
print(V_grid)

# Arrow direction dictionary
action_arrows = {
    0: (-1, 0),  # left
    1: (0, 1),   # down
    2: (1, 0),   # right
    3: (0, -1)   # up
}

# Plot value function and policy
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(V_grid, cmap='coolwarm', interpolation='nearest')
plt.colorbar(im, ax=ax, label="Value V(s)")

for y in range(8):
    for x in range(8):
        action = policy_grid[y, x]
        dx, dy = action_arrows[action]
        ax.arrow(x, y, dx * 0.3, -dy * 0.3, head_width=0.2, head_length=0.2, fc='k', ec='k')
        ax.text(x, y, f"{V_grid[y, x]:.2f}", ha='center', va='center', color='white', fontsize=7)

ax.set_title("Q-Learning: Policy Arrows and Value Function (FrozenLake 8x8)")
ax.axis('off')
plt.show()

# Plot reward trend
plt.figure(figsize=(10, 4))
plt.plot(rewards, color='blue', alpha=0.4, label="Episode reward")

# Smooth running average (window=5000)
window_size = 5000
if len(rewards) >= window_size:
    running_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size - 1, len(rewards)), running_avg, color='red', label=f"Running avg ({window_size})")

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Episode Reward Over Time (Q-Learning)")
plt.legend()
plt.grid(True)
plt.show()