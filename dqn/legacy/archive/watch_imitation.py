"""
Watch your imitation agent play Snake!

Loads a trained model and runs it.
"""

import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

pygame.init()

# ==========================================
# CONFIGURATION
# ==========================================
BLOCK_SIZE = 40
GRID_W, GRID_H = 10, 10
SPEED = 10  # Watchable speed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# MODELS (same as train_imitation.py)
# ==========================================

class ImitationCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.flatten_size = 64 * 8 * 8
        self.fc1 = nn.Linear(self.flatten_size + 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 3)
    
    def forward(self, x, scalars):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.cat((x, scalars), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=device))


class ImitationMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(11, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 3)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=device))


# ==========================================
# GAME ENVIRONMENT
# ==========================================

class SnakeGame:
    def __init__(self):
        self.w = GRID_W * BLOCK_SIZE
        self.h = GRID_H * BLOCK_SIZE + 80
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake - Imitation Agent')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('arial', 20)
        self.reset()
    
    def reset(self):
        self.direction = 1
        self.head = [5, 5]
        self.snake = [self.head[:], [4, 5], [3, 5]]
        self.score = 0
        self.frame_iteration = 0
        self._place_food()
        return self.get_state_cnn(), self.get_scalar_vec(), self.get_state_mlp()
    
    def _place_food(self):
        while True:
            self.food = [random.randint(0, GRID_W-1), random.randint(0, GRID_H-1)]
            if self.food not in self.snake:
                break
    
    def get_state_cnn(self):
        state = np.zeros((4, GRID_H + 2, GRID_W + 2), dtype=np.float32)
        def to_padded(pt): return int(pt[1] + 1), int(pt[0] + 1)
        
        state[3, 0, :] = 1.0; state[3, -1, :] = 1.0
        state[3, :, 0] = 1.0; state[3, :, -1] = 1.0
        
        fr, fc = to_padded(self.food)
        state[1, fr, fc] = 1.0
        
        hr, hc = to_padded(self.head)
        state[0, hr, hc] = 1.0
        
        L = len(self.snake)
        for i, pt in enumerate(reversed(self.snake[1:])):
            if 0 <= pt[0] < GRID_W and 0 <= pt[1] < GRID_H:
                pr, pc = to_padded(pt)
                state[2, pr, pc] = (i + 1) / L
        
        return state
    
    def get_scalar_vec(self):
        dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        dx, dy = dirs[self.direction]
        norm_len = len(self.snake) / (GRID_W * GRID_H)
        return np.array([norm_len, dx, dy], dtype=np.float32)
    
    def get_state_mlp(self):
        head = self.head
        dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        dx, dy = dirs[self.direction]
        
        point_straight = [head[0] + dx, head[1] + dy]
        right_dir = (self.direction + 1) % 4
        rdx, rdy = dirs[right_dir]
        point_right = [head[0] + rdx, head[1] + rdy]
        left_dir = (self.direction - 1) % 4
        ldx, ldy = dirs[left_dir]
        point_left = [head[0] + ldx, head[1] + ldy]
        
        def is_collision(pt):
            if pt[0] < 0 or pt[0] >= GRID_W or pt[1] < 0 or pt[1] >= GRID_H:
                return True
            if pt in self.snake[1:]:
                return True
            return False
        
        state = [
            is_collision(point_straight),
            is_collision(point_right),
            is_collision(point_left),
            self.direction == 0,
            self.direction == 1,
            self.direction == 2,
            self.direction == 3,
            self.food[0] < head[0],
            self.food[0] > head[0],
            self.food[1] < head[1],
            self.food[1] > head[1],
        ]
        return np.array(state, dtype=np.float32)
    
    def step(self, action):
        self.frame_iteration += 1
        
        if action == 0:
            self.direction = (self.direction - 1) % 4
        elif action == 2:
            self.direction = (self.direction + 1) % 4
        
        dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        dx, dy = dirs[self.direction]
        self.head = [self.head[0] + dx, self.head[1] + dy]
        
        # Check collision
        game_over = False
        if (self.head[0] < 0 or self.head[0] >= GRID_W or 
            self.head[1] < 0 or self.head[1] >= GRID_H):
            game_over = True
            return game_over, self.score
        
        if self.head in self.snake[:-1]:
            game_over = True
            return game_over, self.score
        
        self.snake.insert(0, self.head[:])
        
        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()
        
        # Timeout
        if self.frame_iteration > 100 * len(self.snake):
            game_over = True
        
        return game_over, self.score
    
    def render(self, action_probs=None):
        self.display.fill((0, 0, 0))
        
        # Grid
        for x in range(GRID_W + 1):
            pygame.draw.line(self.display, (50, 50, 50),
                           (x * BLOCK_SIZE, 0),
                           (x * BLOCK_SIZE, GRID_H * BLOCK_SIZE))
        for y in range(GRID_H + 1):
            pygame.draw.line(self.display, (50, 50, 50),
                           (0, y * BLOCK_SIZE),
                           (GRID_W * BLOCK_SIZE, y * BLOCK_SIZE))
        
        # Snake
        for i, pt in enumerate(self.snake):
            rect = pygame.Rect(pt[0]*BLOCK_SIZE, pt[1]*BLOCK_SIZE,
                             BLOCK_SIZE-1, BLOCK_SIZE-1)
            color = (0, 200, 255) if i == 0 else (0, 255, 0)
            pygame.draw.rect(self.display, color, rect)
        
        # Food
        food_rect = pygame.Rect(self.food[0]*BLOCK_SIZE, self.food[1]*BLOCK_SIZE,
                               BLOCK_SIZE-1, BLOCK_SIZE-1)
        pygame.draw.rect(self.display, (255, 0, 0), food_rect)
        
        # Info
        info_y = GRID_H * BLOCK_SIZE + 10
        score_text = self.font.render(f'Score: {self.score}', True, (255, 255, 255))
        self.display.blit(score_text, (10, info_y))
        
        # Show action probabilities
        if action_probs is not None:
            probs_text = self.font.render(
                f'L: {action_probs[0]:.2f} | S: {action_probs[1]:.2f} | R: {action_probs[2]:.2f}',
                True, (200, 200, 200)
            )
            self.display.blit(probs_text, (10, info_y + 25))
        
        controls_text = self.font.render('Press R to restart, Q to quit', True, (150, 150, 150))
        self.display.blit(controls_text, (10, info_y + 50))
        
        pygame.display.flip()
        self.clock.tick(SPEED)


def run_agent(model_path, model_type='cnn', num_games=10):
    """Run the imitation agent."""
    
    # Load model
    if model_type == 'cnn':
        model = ImitationCNN().to(device)
    else:
        model = ImitationMLP().to(device)
    
    model.load(model_path)
    model.eval()
    
    print(f"Loaded {model_type.upper()} model from {model_path}")
    
    game = SnakeGame()
    scores = []
    
    running = True
    games_played = 0
    
    state_cnn, scalar_vec, state_mlp = game.reset()
    
    while running and games_played < num_games:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_r:
                    state_cnn, scalar_vec, state_mlp = game.reset()
        
        # Get action from model
        with torch.no_grad():
            if model_type == 'cnn':
                s = torch.tensor(state_cnn, dtype=torch.float).unsqueeze(0).to(device)
                sc = torch.tensor(scalar_vec, dtype=torch.float).unsqueeze(0).to(device)
                logits = model(s, sc)
            else:
                s = torch.tensor(state_mlp, dtype=torch.float).unsqueeze(0).to(device)
                logits = model(s)
            
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            action = logits.argmax(dim=1).item()
        
        # Step
        game_over, score = game.step(action)
        
        # Update states
        state_cnn = game.get_state_cnn()
        scalar_vec = game.get_scalar_vec()
        state_mlp = game.get_state_mlp()
        
        # Render
        game.render(action_probs=probs)
        
        if game_over:
            scores.append(score)
            games_played += 1
            print(f"Game {games_played}: Score = {score}")
            
            if games_played < num_games:
                state_cnn, scalar_vec, state_mlp = game.reset()
    
    pygame.quit()
    
    if scores:
        print(f"\n{'='*40}")
        print(f"Games: {len(scores)}")
        print(f"Average: {np.mean(scores):.1f}")
        print(f"Best: {max(scores)}")
        print(f"Worst: {min(scores)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Watch imitation agent play')
    parser.add_argument('--model', type=str, default='imitation_cnn_best.pth',
                        help='Path to model file')
    parser.add_argument('--type', type=str, default='cnn', choices=['cnn', 'mlp'],
                        help='Model type')
    parser.add_argument('--games', type=int, default=10,
                        help='Number of games to play')
    
    args = parser.parse_args()
    
    run_agent(args.model, model_type=args.type, num_games=args.games)