import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

# ==========================================
# CONFIGURATION
# ==========================================
BLOCK_SIZE = 40
GRID_W, GRID_H = 10, 10
SPEED = 30  # Slower speed for watching
MODEL_PATH = "model_latest.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. THE DUELING NETWORK (Must match training)
# ==========================================
class DuelingDQN(nn.Module):
    def __init__(self, output_size):
        super(DuelingDQN, self).__init__()
        # Input: (4, 12, 12) -> 4 Channels, 12x12
        
        # Feature Extractor (CNN)
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=0) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        
        # Flatten size: 64 channels * 8 * 8 spatial = 4096 
        self.flatten_size = 64 * 8 * 8 
        
        # Dueling Streams
        self.input_dim = self.flatten_size + 3

        # Value Stream (V)
        self.val_fc1 = nn.Linear(self.input_dim, 128)
        self.val_fc2 = nn.Linear(128, 1)

        # Advantage Stream (A)
        self.adv_fc1 = nn.Linear(self.input_dim, 128)
        self.adv_fc2 = nn.Linear(128, output_size) # 3 Actions

    def forward(self, x, scalars):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1) # Flatten

        x = torch.cat((x, scalars), dim=1)

        val = F.relu(self.val_fc1(x))
        val = self.val_fc2(val)

        adv = F.relu(self.adv_fc1(x))
        adv = self.adv_fc2(adv)

        return val + (adv - adv.mean(dim=1, keepdim=True))

# ==========================================
# 2. GAME ENVIRONMENT (With UI & Death Reasons)
# ==========================================
class SnakeGameWatch:
    def __init__(self):
        self.w = GRID_W * BLOCK_SIZE
        self.h = GRID_H * BLOCK_SIZE
        
        pygame.init()
        self.display = pygame.display.set_mode((self.w, self.h + 100)) # Extra space for UI
        pygame.display.set_caption('Snake Agent Watcher')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 20)
        self.big_font = pygame.font.SysFont('Arial', 30)
        
        self.reset()

    def reset(self):
        self.direction = (1, 0) # Right
        self.head = np.array([5, 5]) 
        self.snake = [self.head, self.head - [1, 0], self.head - [2, 0]]
        self.score = 0
        self.frame_iteration = 0
        self.death_reason = None
        self._place_food()
        return self.get_state_bundle()

    def _place_food(self):
        while True:
            self.food = np.array([random.randint(0, GRID_W-1), random.randint(0, GRID_H-1)])
            if not any(np.array_equal(self.food, s) for s in self.snake):
                break

    def step(self, action_idx):
        self.frame_iteration += 1
        
        # Event Handling (Quit & Reset)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.death_reason = "RESET"
                    return True, self.score

        # Determine Move
        clock_wise = [(0,-1), (1,0), (0,1), (-1,0)] # Up, Right, Down, Left
        
        curr_idx = -1
        for i, d in enumerate(clock_wise):
            if d == tuple(self.direction):
                curr_idx = i
                break
        
        if action_idx == 0: # Left
            new_dir_idx = (curr_idx - 1) % 4
        elif action_idx == 2: # Right
            new_dir_idx = (curr_idx + 1) % 4
        else: # Straight
            new_dir_idx = curr_idx
            
        self.direction = np.array(clock_wise[new_dir_idx])
        self.head = self.head + self.direction

        # Check Collisions
        terminated = False
        
        # Wall Collision
        if (self.head[0] < 0 or self.head[0] >= GRID_W or 
            self.head[1] < 0 or self.head[1] >= GRID_H):
            self.death_reason = "HIT WALL"
            return True, self.score

        # Self Collision
        for part in self.snake[:-1]:
            if np.array_equal(self.head, part):
                self.death_reason = "HIT SELF"
                return True, self.score

        # Update Snake
        self.snake.insert(0, self.head)
        
        # Check Food
        if np.array_equal(self.head, self.food):
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()
            
        # Check Timeout
        if self.frame_iteration > 100 * len(self.snake):
            self.death_reason = "TIMEOUT"
            return True, self.score
            
        return False, self.score

    def get_state_bundle(self):
        state = np.zeros((4, GRID_H + 2, GRID_W + 2), dtype=np.float32)
        def to_padded(pt): return int(pt[1]+1), int(pt[0]+1)
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
        norm_len = len(self.snake) / (GRID_W * GRID_H)
        scalar = np.array([norm_len, self.direction[0], self.direction[1]], dtype=np.float32)
        return state, scalar
    
    def render(self):
        self.display.fill((0,0,0))
        
        # Draw Game Area
        for pt in self.snake:
            rect = pygame.Rect(pt[0]*BLOCK_SIZE, pt[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
            pygame.draw.rect(self.display, (0, 255, 0), rect)
        
        h_rect = pygame.Rect(self.head[0]*BLOCK_SIZE, self.head[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
        pygame.draw.rect(self.display, (0, 200, 255), h_rect)

        f_rect = pygame.Rect(self.food[0]*BLOCK_SIZE, self.food[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
        pygame.draw.rect(self.display, (255, 0, 0), f_rect)
        
        # Draw UI Panel
        ui_y = self.h
        pygame.draw.rect(self.display, (30, 30, 30), (0, ui_y, self.w, 100))
        
        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.display.blit(score_text, (10, ui_y + 10))
        
        reset_hint = self.font.render("Press R to Reset", True, (150, 150, 150))
        self.display.blit(reset_hint, (self.w - 150, ui_y + 10))
        
        if self.death_reason:
            reason_text = self.big_font.render(f"{self.death_reason}!", True, (255, 50, 50))
            self.display.blit(reason_text, (10, ui_y + 40))
            
            restart_text = self.font.render("Press SPACE to Restart", True, (200, 200, 200))
            self.display.blit(restart_text, (10, ui_y + 75))

        pygame.display.flip()
        self.clock.tick(SPEED)

# ==========================================
# 3. MAIN LOOP
# ==========================================
def main():
    # Load Model
    try:
        model = DuelingDQN(3).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print(f"Loaded {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Error: {MODEL_PATH} not found. Make sure you have trained the model first.")
        return

    game = SnakeGameWatch()
    
    while True:
        # 1. Play Episode
        state_img, scalar_vec = game.reset()
        done = False
        
        while not done:
            # AI Move
            with torch.no_grad():
                s = torch.tensor(state_img, dtype=torch.float).unsqueeze(0).to(device)
                sc = torch.tensor(scalar_vec, dtype=torch.float).unsqueeze(0).to(device)
                action = model(s, sc).argmax().item()
            
            done, score = game.step(action)
            state_img, scalar_vec = game.get_state_bundle()
            game.render()
            
        # 2. Game Over Screen (Wait for Restart)
        waiting = True
        if game.death_reason == "RESET":
            waiting = False
            
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE or event.key == pygame.K_RETURN:
                        waiting = False
            
            # Keep rendering the game over state
            game.render()

if __name__ == '__main__':
    main()
