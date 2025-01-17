import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import random

#Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 400
DINO_WIDTH, DINO_HEIGHT = 40, 40
OBSTACLE_WIDTH, OBSTACLE_HEIGHT = 20, 40
FLYING_OBSTACLE_WIDTH, FLYING_OBSTACLE_HEIGHT = 30, 20
GROUND_HEIGHT = 300
FONT_SIZE = 24
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
FPS = 60


class DinoGame(gym.Env):
    def __init__(self):
        super(DinoGame, self).__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=SCREEN_WIDTH, shape=(30,), dtype=np.float32)

        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Google Dino Game")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, FONT_SIZE)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.dino_y = GROUND_HEIGHT - DINO_HEIGHT
        self.dino_velocity = 0
        self.is_jumping = False

        # Ground obstacles
        self.ground_obstacles = [{"x": SCREEN_WIDTH + i * 300, "speed": random.randint(4, 8)} for i in range(3)]

        # Flying obstacles
        self.flying_obstacles = [{"x": SCREEN_WIDTH + i * 400, "y": random.randint(150, 250), "speed": random.randint(4, 8)} for i in range(2)]

        self.reward = 0
        self.score = 0  # Continuous score
        self.state = self._get_state()
        return self.state, {}

    def _get_state(self):
        state = np.zeros(30)

        # need to store obstacle1_x, obstacle1_speed, obstacle2_x, obstacle2_speed, etc.
        # need to store bird1_x, bird1_y, bird1_speed, bird2_x, bird2_y, bird2_speed

        # first pos in array is dino_y, pos 1 - 15 contain info about obstacles, pos 16 - 29 contain info about birds
        state[0] = self.dino_y

        i = 1
        for obstacle in self.ground_obstacles:
            state[i] = obstacle["x"]
            i += 1
            state[i] = obstacle["speed"]
            i += 1
            if i > 15:
                break

        i = 16
        for flying in self.flying_obstacles:
            state[i] = flying["x"]
            i += 1
            state[i] = flying["y"]
            i += 1
            state[i] = flying["speed"]
            i += 1
            if i > 29:
              break

        return state

    def step(self, action):
        if action == 1 and not self.is_jumping:
            self.is_jumping = True
            self.dino_velocity = -20

        if self.is_jumping:
            self.dino_y += self.dino_velocity
            self.dino_velocity += 1
            if self.dino_y >= GROUND_HEIGHT - DINO_HEIGHT:
                self.dino_y = GROUND_HEIGHT - DINO_HEIGHT
                self.is_jumping = False

        for obstacle in self.ground_obstacles:
            obstacle["x"] -= obstacle["speed"]
            if obstacle["x"] < 0:
                obstacle["x"] = SCREEN_WIDTH + random.randint(100, 300)
                obstacle["speed"] = random.randint(4, 8)
                self.reward += 1  # Reward for passing obstacle

        for obstacle in self.flying_obstacles:
            obstacle["x"] -= obstacle["speed"]
            if obstacle["x"] < 0:
                obstacle["x"] = SCREEN_WIDTH + random.randint(300, 500)
                obstacle["y"] = random.randint(150, 250)
                obstacle["speed"] = random.randint(4, 8)
                self.reward += 1  # Reward for passing obstacle

        # Increment score continuously
        self.score += 8 / FPS  # Adjust based on frame rate

        done = False
        for obstacle in self.ground_obstacles:
            if (
                obstacle["x"] < 50 + DINO_WIDTH and
                obstacle["x"] + OBSTACLE_WIDTH > 50 and
                self.dino_y + DINO_HEIGHT > GROUND_HEIGHT - OBSTACLE_HEIGHT
            ):
                done = True

        for obstacle in self.flying_obstacles:
            if (
                obstacle["x"] < 50 + DINO_WIDTH and
                obstacle["x"] + FLYING_OBSTACLE_WIDTH > 50 and
                self.dino_y < obstacle["y"] + FLYING_OBSTACLE_HEIGHT and
                self.dino_y + DINO_HEIGHT > obstacle["y"]
            ):
                done = True

        self.state = self._get_state()
        reward = 1 if not done else -100
        return self.state, reward, done, False, {}

    def render(self, mode="human"):
        self.screen.fill(WHITE)
        pygame.draw.line(self.screen, BLACK, (0, GROUND_HEIGHT), (SCREEN_WIDTH, GROUND_HEIGHT), 2)
        pygame.draw.rect(self.screen, BLACK, (50, self.dino_y, DINO_WIDTH, DINO_HEIGHT))

        # Draw ground obstacles
        for obstacle in self.ground_obstacles:
            pygame.draw.rect(self.screen, RED, (obstacle["x"], GROUND_HEIGHT - OBSTACLE_HEIGHT, OBSTACLE_WIDTH, OBSTACLE_HEIGHT))

        # Draw flying obstacles
        for obstacle in self.flying_obstacles:
            pygame.draw.rect(self.screen, BLUE, (obstacle["x"], obstacle["y"], FLYING_OBSTACLE_WIDTH, FLYING_OBSTACLE_HEIGHT))

        score_text = self.font.render(f"Score: {int(self.score)}", True, BLACK)
        self.screen.blit(score_text, (10, 10))
        pygame.display.flip()
        self.clock.tick(FPS)

        return self.screen

    def close(self):
        pygame.quit()

