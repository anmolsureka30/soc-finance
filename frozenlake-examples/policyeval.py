import numpy as np 
import gym
import matplotlib.pyplot as plt


env = gym.make("FrozenLake-v1" ,map_name = "8x8", is_slippery= True , render_mode=None)
gamma = 0.99
theta = 1e-6
n_states = env.observation_space.n
n_actions = env.action_space.n

V = np.zeros(n_states)
policy=np.ones([n_states,n_actions])/n_actions

def policy_evaluation(policy , env , gamma=0.99 , theta=1e-6):
    V=np.zeros(env.observation_space.n)
    while True:
        delta = 0
        for s in range(env.observation_space.n):
            v=0
            for a , action_prob in enumerate(policy[s]):
                for transition in env.P[s][a]:
                    prob, next_state, reward, done = transition
                    v += action_prob*prob*(reward+gamma*V[next_state])
            delta = max(delta, np.abs(V[s]-v))
            V[s] = v
        if delta < theta:
            break
    return V
V = policy_evaluation(policy,env,gamma,theta)

V_grid = V.reshape(8, 8)


plt.figure(figsize=(8, 6))
plt.imshow(V_grid, cmap='coolwarm', interpolation='nearest')
plt.colorbar(label='State Value (V)')

# Annotate with numeric values
for i in range(8):
    for j in range(8):
        plt.text(j, i, f"{V_grid[i, j]:.2f}", ha='center', va='center', color='black', fontsize=8)

plt.title("State Value Function Heatmap (FrozenLake 8x8)")
plt.axis('off')
plt.show()

## These numbers tell me how much total future reward
#  I can expect if I stand in this cell and just wander around randomly forever (policy π).
#  The closer to goal, the higher the number. In holes or dangerous paths, almost zero.
