Game Explanation
This project is a modified version of the classic "Google Dino Game," with added features to make the game more engaging and challenging. The main changes to the original game include the introduction of moving ground obstacles and flying obstacles. Here's a breakdown:

Game Features
Ground Obstacles

Obstacles on the ground move leftward at random speeds.
Once they exit the screen, they reappear on the right at a random position with a new speed.
Flying Obstacles

Flying obstacles appear at random heights and move leftward at varying speeds.
These obstacles make the game more dynamic by requiring precise timing to jump.
Score Mechanism

Players earn points continuously as time progresses and for successfully avoiding obstacles.
Game Termination

The game ends if the dinosaur collides with any obstacle (ground or flying).
Customizable

The game state and rewards are structured to support reinforcement learning agents like DQN for training.
State Representation
The game state is represented as a 30-dimensional vector:

Position 0: The vertical position of the dino (dino_y).
Positions 1-15: Information about the ground obstacles (x-coordinate, speed).
Positions 16-29: Information about the flying obstacles (x-coordinate, y-coordinate, speed).
This state vector is designed to provide all necessary information for an agent to make decisions.

Action Space
Discrete(2): The agent (or player) has two possible actions:
0: Do nothing (keep running).
1: Jump.
Reward Function
Positive Reward:
+1 for successfully passing each obstacle.
Continuous reward is earned over time, proportional to the score.

Negative Reward:
-100 if the dinosaur collides with any obstacle, ending the game.

Instructions to Run the Game
Clone or download this repository to your local machine.
Run game.py to play the game manually.
Instructions to Train a Reinforcement Learning (DQN) Agent
Clone or download this repository.
Run train_agent.py to train an RL agent using the stable-baselines3 library.
Contributions
Created at TUMO.
Utilized Gymnasium for the game environment and stable-baselines3 for RL training.
This project is both a playable game and a learning environment for reinforcement learning experiments. It allows customization for testing various RL algorithms. 
