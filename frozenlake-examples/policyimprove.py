import numpy as np
import gym
import matplotlib.pyplot as plt
from matplotlib import animation

env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)
gamma = 0.99
theta = 1e-6

n_states = env.observation_space.n
n_actions = env.action_space.n
policy = np.ones([n_states, n_actions]) / n_actions

# Store value grids for animation
V_history = []

def policy_evaluation(policy, env, gamma=0.99, theta=1e-6):
    V = np.zeros(env.observation_space.n)
    local_history = []
    while True:
        delta = 0
        V_copy = V.copy()
        for s in range(env.observation_space.n):
            v = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    v += action_prob * prob * (reward + gamma * V[next_state])
            delta = max(delta, np.abs(V[s] - v))
            V[s] = v
        local_history.append(V.copy().reshape(8, 8))  # Save snapshot after each small update
        if delta < theta:
            break
    return V, local_history

def policy_improvement(V, env, gamma=0.99):
    policy = np.zeros([env.observation_space.n, env.action_space.n])
    for s in range(env.observation_space.n):
        q_values = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for prob, next_state, reward, done in env.P[s][a]:
                q_values[a] += prob * (reward + gamma * V[next_state])
        best_action = np.argmax(q_values)
        policy[s, best_action] = 1.0
    return policy

# Policy iteration loop
is_policy_stable = False
iteration = 0
while not is_policy_stable:
    iteration += 1
    V, local_history = policy_evaluation(policy, env, gamma, theta)
    V_history.extend(local_history)  # Append all inner V updates
    new_policy = policy_improvement(V, env, gamma)
    if np.array_equal(new_policy, policy):
        is_policy_stable = True
        print(f"Policy converged after {iteration} iterations.")
    policy = new_policy.copy()

print("Final optimal state value function V(s):")
print(V.reshape(8, 8))

# Animation setup
fig, ax = plt.subplots(figsize=(8, 6))

def update(frame):
    ax.clear()
    V_grid = V_history[frame]
    im = ax.imshow(V_grid, cmap='coolwarm', interpolation='nearest')
    ax.set_title(f"Value Iteration Step {frame+1}")
    ax.axis('off')
    
    # Add value text on each cell
    for i in range(8):
        for j in range(8):
            ax.text(j, i, f"{V_grid[i, j]:.2f}", ha='center', va='center', color='black', fontsize=6)
    return [im]

anim = animation.FuncAnimation(fig, update, frames=len(V_history), interval=100, blit=False)

plt.show()

