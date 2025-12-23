"""
Snake Game - Human Play Recorder
Records (state, action) pairs as you play for imitation learning.

Controls:
    Arrow keys or WASD to move
    R to restart after death
    Q to quit and save

Data is saved to 'human_plays.npz'
"""

import pygame
import random
import numpy as np
import os
from datetime import datetime

pygame.init()

# ==========================================
# CONFIGURATION
# ==========================================
BLOCK_SIZE = 40
GRID_W, GRID_H = 10, 10
SPEED = 10  # Comfortable human speed

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
DARK_GREEN = (0, 200, 0)
CYAN = (0, 200, 255)
RED = (255, 0, 0)
GRAY = (50, 50, 50)

# ==========================================
# GAME WITH RECORDING
# ==========================================
class SnakeGameHuman:
    def __init__(self):
        self.w = GRID_W * BLOCK_SIZE
        self.h = GRID_H * BLOCK_SIZE + 60  # Extra space for info
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake - Recording Your Plays')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('arial', 20)
        
        # Recording storage
        self.all_states = []
        self.all_actions = []
        self.games_played = 0
        self.total_frames = 0
        
        self.reset()
    
    def reset(self):
        self.direction = 1  # 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        self.head = [5, 5]
        self.snake = [self.head[:], [4, 5], [3, 5]]
        self.score = 0
        self.frame_iteration = 0
        self._place_food()
        self.game_over = False
        self.waiting_for_input = True
    
    def _place_food(self):
        while True:
            self.food = [random.randint(0, GRID_W-1), random.randint(0, GRID_H-1)]
            if self.food not in self.snake:
                break
    
    def get_state_cnn(self):
        """
        Returns the same 4-channel state as your RL agent.
        Shape: (4, 12, 12)
        """
        state = np.zeros((4, GRID_H + 2, GRID_W + 2), dtype=np.float32)
        
        def to_padded(pt): 
            return int(pt[1] + 1), int(pt[0] + 1)
        
        # Channel 3: Walls
        state[3, 0, :] = 1.0
        state[3, -1, :] = 1.0
        state[3, :, 0] = 1.0
        state[3, :, -1] = 1.0
        
        # Channel 1: Food
        fr, fc = to_padded(self.food)
        state[1, fr, fc] = 1.0
        
        # Channel 0: Head
        hr, hc = to_padded(self.head)
        state[0, hr, hc] = 1.0
        
        # Channel 2: Body time-to-clear
        L = len(self.snake)
        for i, pt in enumerate(reversed(self.snake[1:])):
            if 0 <= pt[0] < GRID_W and 0 <= pt[1] < GRID_H:
                pr, pc = to_padded(pt)
                state[2, pr, pc] = (i + 1) / L
        
        return state
    
    def get_state_mlp(self):
        """
        Returns the 11-feature state like the simple MLP agent.
        """
        head = self.head
        
        dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # UP, RIGHT, DOWN, LEFT
        dx, dy = dirs[self.direction]
        
        # Points around head
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
    
    def get_scalar_vec(self):
        """Direction scalars for CNN model."""
        dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        dx, dy = dirs[self.direction]
        norm_len = len(self.snake) / (GRID_W * GRID_H)
        return np.array([norm_len, dx, dy], dtype=np.float32)
    
    def key_to_action(self, key):
        """
        Convert keypress to relative action.
        Returns: 0=left turn, 1=straight, 2=right turn
        """
        dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # UP, RIGHT, DOWN, LEFT
        
        # Map key to absolute direction
        key_to_dir = {
            pygame.K_UP: 0, pygame.K_w: 0,
            pygame.K_RIGHT: 1, pygame.K_d: 1,
            pygame.K_DOWN: 2, pygame.K_s: 2,
            pygame.K_LEFT: 3, pygame.K_a: 3,
        }
        
        if key not in key_to_dir:
            return None
        
        target_dir = key_to_dir[key]
        
        # Can't reverse
        if (target_dir + 2) % 4 == self.direction:
            return 1  # Treat as straight
        
        # Calculate relative action
        if target_dir == self.direction:
            return 1  # Straight
        elif target_dir == (self.direction + 1) % 4:
            return 2  # Right turn
        elif target_dir == (self.direction - 1) % 4:
            return 0  # Left turn
        else:
            return 1  # Default straight
    
    def step(self, action):
        """
        action: 0=left, 1=straight, 2=right
        """
        self.frame_iteration += 1
        
        # Update direction
        if action == 0:  # Left
            self.direction = (self.direction - 1) % 4
        elif action == 2:  # Right
            self.direction = (self.direction + 1) % 4
        
        # Move head
        dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        dx, dy = dirs[self.direction]
        self.head = [self.head[0] + dx, self.head[1] + dy]
        
        # Check collision
        if (self.head[0] < 0 or self.head[0] >= GRID_W or 
            self.head[1] < 0 or self.head[1] >= GRID_H):
            self.game_over = True
            return
        
        if self.head in self.snake[:-1]:
            self.game_over = True
            return
        
        # Update snake
        self.snake.insert(0, self.head[:])
        
        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()
    
    def render(self, skip_delay=False):
        self.display.fill(BLACK)
        
        # Draw grid lines (subtle)
        for x in range(GRID_W + 1):
            pygame.draw.line(self.display, GRAY, 
                           (x * BLOCK_SIZE, 0), 
                           (x * BLOCK_SIZE, GRID_H * BLOCK_SIZE))
        for y in range(GRID_H + 1):
            pygame.draw.line(self.display, GRAY,
                           (0, y * BLOCK_SIZE),
                           (GRID_W * BLOCK_SIZE, y * BLOCK_SIZE))
        
        # Draw snake
        for i, pt in enumerate(self.snake):
            rect = pygame.Rect(pt[0]*BLOCK_SIZE, pt[1]*BLOCK_SIZE, 
                             BLOCK_SIZE-1, BLOCK_SIZE-1)
            if i == 0:  # Head
                pygame.draw.rect(self.display, CYAN, rect)
            else:
                pygame.draw.rect(self.display, GREEN, rect)
        
        # Draw food
        food_rect = pygame.Rect(self.food[0]*BLOCK_SIZE, self.food[1]*BLOCK_SIZE,
                               BLOCK_SIZE-1, BLOCK_SIZE-1)
        pygame.draw.rect(self.display, RED, food_rect)
        
        # Draw info bar
        info_y = GRID_H * BLOCK_SIZE + 10
        score_text = self.font.render(f'Score: {self.score}', True, WHITE)
        games_text = self.font.render(f'Games: {self.games_played}', True, WHITE)
        frames_text = self.font.render(f'Frames: {self.total_frames}', True, WHITE)
        
        self.display.blit(score_text, (10, info_y))
        self.display.blit(games_text, (150, info_y))
        self.display.blit(frames_text, (280, info_y))
        
        if self.game_over:
            go_text = self.font.render('GAME OVER - Press R to restart, Q to quit & save', 
                                       True, RED)
            self.display.blit(go_text, (10, info_y + 25))
        
        pygame.display.flip()
        
        if not skip_delay:
            self.clock.tick(SPEED)
    
    def save_data(self):
        """Save all recorded data."""
        if len(self.all_states) == 0:
            print("No data to save!")
            return
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'human_plays_{timestamp}.npz'
        
        np.savez(filename,
                 states=np.array(self.all_states),
                 actions=np.array(self.all_actions),
                 games=self.games_played,
                 frames=self.total_frames)
        
        print(f"\nSaved {len(self.all_states)} frames from {self.games_played} games to {filename}")
    
    def get_action_from_keys(self):
        """
        Poll keyboard state for responsive input.
        Returns relative action: 0=left, 1=straight, 2=right
        """
        keys = pygame.key.get_pressed()
        
        # Determine target absolute direction from keys
        target_dir = None
        
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            target_dir = 0  # UP
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            target_dir = 1  # RIGHT
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            target_dir = 2  # DOWN
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            target_dir = 3  # LEFT
        
        if target_dir is None:
            return None  # No input
        
        # Can't reverse (180 degree turn)
        if (target_dir + 2) % 4 == self.direction:
            return None  # Ignore reverse
        
        # Convert absolute direction to relative action
        if target_dir == self.direction:
            return 1  # Straight
        elif target_dir == (self.direction + 1) % 4:
            return 2  # Right turn
        elif target_dir == (self.direction - 1) % 4:
            return 0  # Left turn
        else:
            return 1  # Fallback to straight
    
    def collect_input(self, duration_ms):
        """
        Collect input over a time window.
        Returns the last valid directional input, or 1 (straight) if none.
        """
        last_action = 1  # Default straight
        start_time = pygame.time.get_ticks()
        
        while pygame.time.get_ticks() - start_time < duration_ms:
            # Must pump events for key state to update
            pygame.event.pump()
            
            action = self.get_action_from_keys()
            if action is not None:
                last_action = action
            
            # Small sleep to not burn CPU
            pygame.time.wait(5)
        
        return last_action
    
    def run(self):
        """Main game loop."""
        print("\n" + "="*50)
        print("SNAKE - Recording Mode")
        print("="*50)
        print("Controls: Arrow keys or WASD (hold direction)")
        print("R = Restart after death")
        print("Q = Quit and save data")
        print("="*50 + "\n")
        
        running = True
        
        while running:
            # Handle quit/restart events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r and self.game_over:
                        self.games_played += 1
                        self.reset()
            
            # Game logic
            if not self.game_over:
                # Collect input over the frame duration (catches quick presses)
                move_delay_ms = 1000 // SPEED  # e.g., 100ms at SPEED=10
                action = self.collect_input(move_delay_ms)
                
                # Record state BEFORE action
                state_cnn = self.get_state_cnn()
                scalar_vec = self.get_scalar_vec()
                state_mlp = self.get_state_mlp()
                
                # Store combined state
                combined_state = {
                    'cnn': state_cnn,
                    'scalar': scalar_vec,
                    'mlp': state_mlp
                }
                
                self.all_states.append(combined_state)
                self.all_actions.append(action)
                self.total_frames += 1
                
                # Execute action
                self.step(action)
                
                if self.game_over:
                    self.games_played += 1
                    print(f"Game {self.games_played} ended | Score: {self.score} | Total frames: {self.total_frames}")
            
            self.render(skip_delay=True)  # Skip clock.tick since we handled timing
        
        else:
            # Game over - just render and wait for input
            self.render(skip_delay=False)
        
        self.save_data()
        pygame.quit()


if __name__ == '__main__':
    game = SnakeGameHuman()
    game.run()