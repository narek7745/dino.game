from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
import imageio
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from env import DinoGame
import gymnasium as gym
import pygame


class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.rewards = []

    def _on_step(self) -> bool:
        # Log the reward if available
        reward = self.locals.get("rewards", None)
        if reward is not None:
            self.rewards.append(reward)
        return True

    def _on_rollout_end(self):
        # Optionally, log rewards at the end of a rollout
        if len(self.rewards) > 0:
            print(f"Rollout rewards: {sum(self.rewards)}")

gym.envs.registration.register(
    id='Game',
    entry_point=DinoGame
)

env = gym.make("Game")

# Step 2: Set up the DQN model
model = DQN(
    "MlpPolicy",  # Use "CnnPolicy" for pixel-based environments
    env,
    verbose=1,
    learning_rate=0.0001,
    buffer_size=100000,
    batch_size=32,
)

# Step 3: Define a reward logging callback
class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                print(f"Episode reward: {info['episode']['r']}")
        return True

reward_callback = RewardLoggerCallback()

# Step 4: Train the model
# TODO: Experiment with different timesteps to measure training time and performance
timesteps = 800000  # Replace with 10, 100, 1000, etc., for experimentation

model.learn(total_timesteps=timesteps, callback=reward_callback)

# Step 5: Save the model
model_path = "game_model"
model.save(model_path)

print(f"Model saved to {model_path}")

# Load the trained model
model_path = "game_model"

# Initialize the environment
env = gym.make("Game")
obs, _ = env.reset()

frames = []

for _ in range(1000):
    # Get action from the trained model
    action, _ = model.predict(obs, deterministic=True)

    # Take a step in the environment using the action
    obs, reward, done, truncated, info = env.step(action)

    # Render the environment and capture the frame
    screen = env.render()
    frame = pygame.surfarray.array3d(screen)  # Capture Pygame surface as a frame
    frame = frame.swapaxes(0, 1)  # Adjust axes for correct orientation
    frames.append(frame)

    if done:
        break

# Save the frames as a GIF
gif_path = "game.gif"
imageio.mimsave(gif_path, frames, fps=10)

# Display the GIF in Colab or Jupyter
from IPython.display import Image
Image(filename=gif_path)
