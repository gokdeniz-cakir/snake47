import pygame
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

pygame.init()

# ==========================================
# CONFIGURATION
# ==========================================
BLOCK_SIZE = 40
GRID_W, GRID_H = 10, 10
SPEED = 0  # 0 for max speed

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
GAMMA = 0.9

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# SIMPLE Q-NETWORK (MLP)
# ==========================================
class SimpleQNet(nn.Module):
    def __init__(self, input_size=11, hidden_size=256, output_size=3):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(self, path='simple_model.pth'):
        torch.save(self.state_dict(), path)

# ==========================================
# ENVIRONMENT
# ==========================================
class SnakeGame:
    def __init__(self, render_mode=True):
        self.render_mode = render_mode
        if self.render_mode:
            self.display = pygame.display.set_mode((GRID_W * BLOCK_SIZE, GRID_H * BLOCK_SIZE))
            pygame.display.set_caption('Simple Snake Agent')
            self.clock = pygame.time.Clock()
        self.reset()
    
    def reset(self):
        self.direction = 1  # 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        self.head = [5, 5]
        self.snake = [self.head[:], [4, 5], [3, 5]]
        self.score = 0
        self.frame_iteration = 0
        self._place_food()
        return self.get_state()
    
    def _place_food(self):
        while True:
            self.food = [random.randint(0, GRID_W-1), random.randint(0, GRID_H-1)]
            if self.food not in self.snake:
                break
    
    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Wall collision
        if pt[0] < 0 or pt[0] >= GRID_W or pt[1] < 0 or pt[1] >= GRID_H:
            return True
        # Self collision (exclude head itself if checking head)
        if pt in self.snake[1:]:
            return True
        return False
    
    def get_state(self):
        """
        11 hand-crafted features:
        [0-2]: Danger straight, right, left
        [3-6]: Direction (one-hot: up, right, down, left)
        [7-10]: Food direction (left, right, up, down)
        """
        head = self.head
        
        # Points around head
        dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # UP, RIGHT, DOWN, LEFT
        dx, dy = dirs[self.direction]
        
        # Point straight ahead
        point_straight = [head[0] + dx, head[1] + dy]
        
        # Point to the right (clockwise)
        right_dir = (self.direction + 1) % 4
        rdx, rdy = dirs[right_dir]
        point_right = [head[0] + rdx, head[1] + rdy]
        
        # Point to the left (counter-clockwise)
        left_dir = (self.direction - 1) % 4
        ldx, ldy = dirs[left_dir]
        point_left = [head[0] + ldx, head[1] + ldy]
        
        state = [
            # Danger straight, right, left
            self._is_collision(point_straight),
            self._is_collision(point_right),
            self._is_collision(point_left),
            
            # Current direction (one-hot)
            self.direction == 0,  # up
            self.direction == 1,  # right
            self.direction == 2,  # down
            self.direction == 3,  # left
            
            # Food direction
            self.food[0] < head[0],  # food left
            self.food[0] > head[0],  # food right
            self.food[1] < head[1],  # food up
            self.food[1] > head[1],  # food down
        ]
        
        return np.array(state, dtype=np.float32)
    
    def step(self, action):
        """
        action: 0 = straight, 1 = right turn, 2 = left turn
        """
        self.frame_iteration += 1
        
        if self.render_mode:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
        
        # Update direction based on action
        if action == 1:  # right turn
            self.direction = (self.direction + 1) % 4
        elif action == 2:  # left turn
            self.direction = (self.direction - 1) % 4
        # action == 0: straight, no change
        
        # Move head
        dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        dx, dy = dirs[self.direction]
        self.head = [self.head[0] + dx, self.head[1] + dy]
        
        # Check collision
        reward = 0
        game_over = False
        
        if self._is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
        
        # Update snake
        self.snake.insert(0, self.head[:])
        
        # Check food
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        
        return reward, game_over, self.score
    
    def render(self):
        if not self.render_mode:
            return
        
        self.display.fill((0, 0, 0))
        
        for pt in self.snake:
            rect = pygame.Rect(pt[0]*BLOCK_SIZE, pt[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
            pygame.draw.rect(self.display, (0, 255, 0), rect)
        
        head_rect = pygame.Rect(self.head[0]*BLOCK_SIZE, self.head[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
        pygame.draw.rect(self.display, (0, 200, 255), head_rect)
        
        food_rect = pygame.Rect(self.food[0]*BLOCK_SIZE, self.food[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
        pygame.draw.rect(self.display, (255, 0, 0), food_rect)
        
        pygame.display.flip()
        if SPEED > 0:
            self.clock.tick(SPEED)

# ==========================================
# AGENT
# ==========================================
class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = GAMMA
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = SimpleQNet().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.criterion = nn.MSELoss()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def train_short_memory(self, state, action, reward, next_state, done):
        self._train_step(state, action, reward, next_state, done)
    
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = list(self.memory)
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self._train_step(states, actions, rewards, next_states, dones)
    
    def _train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float).to(device)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float).to(device)
        action = torch.tensor(np.array(action), dtype=torch.long).to(device)
        reward = torch.tensor(np.array(reward), dtype=torch.float).to(device)
        
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done,)
        
        # Current Q values
        pred = self.model(state)
        
        # Target Q values
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][action[idx].item()] = Q_new
        
        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()
    
    def get_action(self, state):
        # Epsilon decays with number of games
        self.epsilon = 80 - self.n_games
        
        if random.randint(0, 200) < self.epsilon:
            action = random.randint(0, 2)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device)
            prediction = self.model(state_tensor)
            action = torch.argmax(prediction).item()
        
        return action

# ==========================================
# TRAINING
# ==========================================
def train():
    agent = Agent()
    game = SnakeGame(render_mode=False)  # Set True to watch
    
    scores = []
    record = 0
    
    print(f"Training Simple Snake Agent on {device}")
    print("=" * 50)
    
    while True:
        # Get current state
        state = game.get_state()
        
        # Get action
        action = agent.get_action(state)
        
        # Perform action
        reward, done, score = game.step(action)
        
        # Get new state
        next_state = game.get_state()
        
        # Train short memory (single step)
        agent.train_short_memory(state, action, reward, next_state, done)
        
        # Remember
        agent.remember(state, action, reward, next_state, done)
        
        # Render
        game.render()
        
        if done:
            # Reset and train long memory
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            
            scores.append(score)
            
            if score > record:
                record = score
                agent.model.save()
            
            # Print stats every 10 games
            if agent.n_games % 10 == 0:
                avg_last_100 = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
                print(f"Game {agent.n_games} | Score: {score} | Record: {record} | Avg(100): {avg_last_100:.1f}")

if __name__ == '__main__':
    train()