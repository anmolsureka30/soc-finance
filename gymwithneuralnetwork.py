import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import MlpExtractor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
import numpy as np

class CustomMLP(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU()
        )
    def forward(self, observations):
        return self.net(observations)

policy_kwargs = dict(
    features_extractor_class=CustomMLP,
    features_extractor_kwargs=dict(features_dim=128),
    net_arch=[256, 256]
)

env = gym.make('CartPole-v1', render_mode='human')

model = DQN('MlpPolicy', env, verbose=1, learning_rate=1e-3, buffer_size=50000, learning_starts=10, target_update_interval=100, policy_kwargs=policy_kwargs)

model.learn(total_timesteps=50000)

episode_rewards = []
for episode in range(10):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    state = obs
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        state = obs
        env.render()
    episode_rewards.append(total_reward)
    print(f"Episode {episode+1} Score: {total_reward}, Final State: {state}")

print("Average reward over 10 episodes:", np.mean(episode_rewards))
env.close()