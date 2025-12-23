import sys
import random
import numpy as np
import pygame
import torch
import torch.nn as nn

from snake_env import SnakeEnv, ACTION_SET

# --- Copied ConvDQN definition to ensure standalone execution ---
class CNNDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CNNDQN, self).__init__()
        # input_shape is (3, 10, 10)
        
        self.features = nn.Sequential(
            # Conv 1: Detects edges/corners
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Conv 2: Detects shapes/patterns
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        # Calculate size: 10x10 input remains 10x10 due to padding=1
        # Flatten size = 64 channels * 10 * 10 = 6400
        linear_input_size = 64 * 10 * 10
        
        self.fc = nn.Sequential(
            nn.Linear(linear_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) # Flatten
        return self.fc(x)

CELL_SIZE = 40
FPS = 10

def load_policy(path: str, width: int, height: int):
    # Input shape for CNN: (3, H, W)
    input_shape = (3, height, width)
    num_actions = len(ACTION_SET)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")
    
    policy_net = CNNDQN(input_shape, num_actions).to(device)
    try:
        policy_net.load_state_dict(torch.load(path, map_location=device))
    except RuntimeError as e:
        print(f"Error loading model: {e}")
        print("Make sure you are loading a checkpoint trained with the CNN architecture.")
        sys.exit(1)
        
    policy_net.eval()
    return policy_net, device

def select_greedy_action(policy_net: CNNDQN, device: torch.device, env: SnakeEnv, obs, epsilon: float = 0.0):
    if random.random() < epsilon:
        return random.choice(list(ACTION_SET))
    # obs is (3, H, W) from env
    state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        q_values = policy_net(state)
        return int(torch.argmax(q_values, dim=1).item())

def main():
    width, height = 10, 10  # Must match training dimensions

    # Environment settings
    env = SnakeEnv(
        width=width,
        height=height,
        step_cost=-0.01,
        allow_reverse=False,
        dist_reward_coef=0.1,
        food_reward=10.0,
        death_penalty=-10.0,
        max_steps=width * height * 50,
    )

    checkpoint_path = "dqn_snake_cnn.pt"
    
    try:
        policy_net, device = load_policy(checkpoint_path, width, height)
    except FileNotFoundError:
        print(f"Could not find {checkpoint_path}. Train the agent first with train_dqn.py.")
        sys.exit(1)

    pygame.init()
    screen = pygame.display.set_mode((width * CELL_SIZE, height * CELL_SIZE))
    pygame.display.set_caption("Snake RL Agent (Gradient CNNDQN)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    obs = env.reset()
    done = False
    death_reason = ""
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE and done:
                    obs = env.reset()
                    done = False
                    death_reason = ""

        if not done:
            # Pure greedy evaluation
            action = select_greedy_action(policy_net, device, env, obs, epsilon=0.0)
            obs, reward, done, info = env.step(action)
            if done and "death_reason" in info:
                death_reason = info["death_reason"]

        # draw
        screen.fill((0, 0, 0))

        # grid lines
        for x in range(width):
            pygame.draw.line(
                screen,
                (40, 40, 40),
                (x * CELL_SIZE, 0),
                (x * CELL_SIZE, height * CELL_SIZE),
                1,
            )
        for y in range(height):
            pygame.draw.line(
                screen,
                (40, 40, 40),
                (0, y * CELL_SIZE),
                (width * CELL_SIZE, y * CELL_SIZE),
                1,
            )

        # food
        fx, fy = env.food
        food_rect = pygame.Rect(fx * CELL_SIZE, fy * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, (200, 0, 0), food_rect)

        # snake
        for i, (sx, sy) in enumerate(env.snake):
            color = (0, 220, 0) if i == 0 else (0, 140, 0)
            seg_rect = pygame.Rect(sx * CELL_SIZE, sy * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, color, seg_rect)

        score_surf = font.render(f"Score: {env.score}", True, (255, 255, 255))
        screen.blit(score_surf, (8, 8))

        if done:
            msg = f"Game over ({death_reason}) - SPACE to reset, ESC to quit"
            over_surf = font.render(msg, True, (255, 255, 255))
            screen.blit(over_surf, (8, 32))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()
