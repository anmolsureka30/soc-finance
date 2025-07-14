import gym
import random

env= gym.make('CartPole-v1', render_mode='human')
episodes = 10
for episode in range(1,episodes+1):
    state = env.reset()
    done = False
    score = 0
    while not done:
        action = random.choice([0,1])  # Random action
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        score += reward
        env.render()
    print(f'Episode {episode} Score: {score}')
env.close()
    
    