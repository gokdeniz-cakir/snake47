import pygame
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ==========================================
# CONFIGURATION & HYPERPARAMETERS
# ==========================================
# Physics & Rendering
BLOCK_SIZE = 20
GRID_W, GRID_H = 10, 10
SPEED = 60  # Set to 0 for max speed training

# RL Hyperparameters
MAX_MEMORY = 50_000        # Naive PER limit
BATCH_SIZE = 64
LR = 1e-4
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPS_DECAY_STEPS = 100_000
TARGET_UPDATE_INTERVAL = 2_000

# PER Hyperparameters
ALPHA = 0.6
BETA_START = 0.4
BETA_FRAMES = 500_000
PER_EPS = 1e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. THE DUELING NETWORK
# ==========================================
class DuelingDQN(nn.Module):
    def __init__(self, output_size):
        super(DuelingDQN, self).__init__()
        # Input: (4, 12, 12) -> 4 Channels, 12x12
        
        # Feature Extractor (CNN)
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=0) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        
        # Flatten size: 64 channels * 8 * 8 spatial = 4096 
        # (Note: 12x12 -> minus 2 -> 10x10 -> minus 2 -> 8x8)
        self.flatten_size = 64 * 8 * 8 
        
        # Dueling Streams
        # We inject the 3 scalars (Len, DirX, DirY) here
        self.input_dim = self.flatten_size + 3

        # Value Stream (V)
        self.val_fc1 = nn.Linear(self.input_dim, 128)
        self.val_fc2 = nn.Linear(128, 1)

        # Advantage Stream (A)
        self.adv_fc1 = nn.Linear(self.input_dim, 128)
        self.adv_fc2 = nn.Linear(128, output_size) # 3 Actions

    def forward(self, x, scalars):
        # x: (Batch, 4, 12, 12)
        # scalars: (Batch, 3)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1) # Flatten

        # Concatenate Scalar Context
        x = torch.cat((x, scalars), dim=1)

        # Value stream
        val = F.relu(self.val_fc1(x))
        val = self.val_fc2(val)

        # Advantage stream
        adv = F.relu(self.adv_fc1(x))
        adv = self.adv_fc2(adv)

        # Combine: Q = V + (A - mean(A))
        return val + (adv - adv.mean(dim=1, keepdim=True))

    def save(self, path='model_dueling.pth'):
        torch.save(self.state_dict(), path)

# ==========================================
# 2. NAIVE PRIORITIZED REPLAY BUFFER
# ==========================================
class NaivePERBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.pos = 0

    def add(self, state, scalar, action, reward, next_state, next_scalar, done):
        # Default priority is max priority (ensure it gets sampled at least once)
        max_prio = max(self.priorities) if self.buffer else 1.0
        
        data = (state, scalar, action, reward, next_state, next_scalar, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
            self.priorities.append(max_prio)
        else:
            self.buffer[self.pos] = data
            self.priorities[self.pos] = max_prio
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta):
        N = len(self.buffer)
        if N == 0: return None
        
        # Calculate Probabilities
        prios = np.array(self.priorities)
        probs = prios ** ALPHA
        probs /= probs.sum()

        # Sample Indices
        indices = np.random.choice(N, batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        # Calculate Importance Weights
        total = N
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max() # Normalize
        
        return samples, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = abs(error) + PER_EPS

    def __len__(self):
        return len(self.buffer)

# ==========================================
# 3. ENVIRONMENT
# ==========================================
class SnakeGameAI:
    def __init__(self, render_mode=True):
        self.w = GRID_W * BLOCK_SIZE
        self.h = GRID_H * BLOCK_SIZE
        self.render_mode = render_mode
        
        # Only init display if rendering
        if self.render_mode:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake Dueling PER Agent')
            self.clock = pygame.time.Clock()
        
        self.reset()

    def reset(self):
        # Direction: (dx, dy)
        self.direction = (1, 0) # Right
        self.head = np.array([5, 5]) # Grid coords
        self.snake = [self.head, self.head - [1, 0], self.head - [2, 0]]
        self.score = 0
        self.frame_iteration = 0
        self._place_food()
        return self.get_state_bundle()

    def _place_food(self):
        while True:
            self.food = np.array([random.randint(0, GRID_W-1), random.randint(0, GRID_H-1)])
            # Check if food is in snake
            if not any(np.array_equal(self.food, s) for s in self.snake):
                break

    def step(self, action_idx):
        self.frame_iteration += 1
        
        # 1. Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); quit()

        # 2. Determine Move
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

        # 3. Check Collisions
        terminated = False
        truncated = False
        reward = -0.05 # Standard step penalty
        
        # Wall Collision
        if (self.head[0] < 0 or self.head[0] >= GRID_W or 
            self.head[1] < 0 or self.head[1] >= GRID_H):
            terminated = True
            reward = -10
            return reward, terminated, truncated, self.score

        # Self Collision
        # BUG FIX: Check against snake[:-1] (exclude tail). 
        # If we don't eat, tail moves, so hitting the current tail cell is safe.
        for part in self.snake[:-1]:
            if np.array_equal(self.head, part):
                terminated = True
                reward = -10
                return reward, terminated, truncated, self.score

        # 4. Update Snake
        self.snake.insert(0, self.head)
        
        # 5. Check Food
        if np.array_equal(self.head, self.food):
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
            
        # 6. Check Timeout (Truncated)
        # BUG FIX: Return truncated=True, but do NOT set terminated=True
        if self.frame_iteration > 100 * len(self.snake):
            truncated = True
            # Reward remains -0.05 (or 0), do not penalize heavily or it suicides
            
        return reward, terminated, truncated, self.score

    def get_state_bundle(self):
        # Same as previous version...
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
        if not self.render_mode:
            return # Skip drawing for max speed
            
        self.display.fill((0,0,0))
        for pt in self.snake:
            rect = pygame.Rect(pt[0]*BLOCK_SIZE, pt[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
            pygame.draw.rect(self.display, (0, 255, 0), rect)
        
        h_rect = pygame.Rect(self.head[0]*BLOCK_SIZE, self.head[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
        pygame.draw.rect(self.display, (0, 200, 255), h_rect)

        f_rect = pygame.Rect(self.food[0]*BLOCK_SIZE, self.food[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
        pygame.draw.rect(self.display, (255, 0, 0), f_rect)
        
        pygame.display.flip()
        self.clock.tick(SPEED)

# ==========================================
# 4. TRAINING LOOP
# ==========================================
def evaluate(policy_net, games=5):
    """Runs a few games with epsilon=0 to test true performance."""
    test_game = SnakeGameAI(render_mode=False) # Headless for speed
    total_score = 0
    
    for _ in range(games):
        state_img, scalar_vec = test_game.reset()
        done = False
        while not done:
            with torch.no_grad():
                s = torch.tensor(state_img, dtype=torch.float).unsqueeze(0).to(device)
                sc = torch.tensor(scalar_vec, dtype=torch.float).unsqueeze(0).to(device)
                action = policy_net(s, sc).argmax().item()
            
            _, term, trunc, score = test_game.step(action)
            done = term or trunc
            state_img, scalar_vec = test_game.get_state_bundle()
        total_score += score
    
    return total_score / games

def train():
    pygame.init()
    # SETTINGS
    RENDER_TRAINING = False  # Set False for 100x speedup
    EVAL_INTERVAL = 5000    # Evaluate every 5k steps
    CHECKPOINT_INTERVAL = 10000 
    
    game = SnakeGameAI(render_mode=RENDER_TRAINING)
    
    policy_net = DuelingDQN(3).to(device)
    target_net = DuelingDQN(3).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR)
    memory = NaivePERBuffer(MAX_MEMORY)
    
    # Metrics
    steps_done = 0
    best_eval_score = -float('inf')
    recent_scores = deque(maxlen=100)
    
    print(f"Starting Training on {device}...")
    print(f"Rendering: {RENDER_TRAINING} | Eval Interval: {EVAL_INTERVAL}")

    while True:
        state_img, scalar_vec = game.get_state_bundle()
        
        # Epsilon Decay
        epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                  max(0, (EPS_DECAY_STEPS - steps_done) / EPS_DECAY_STEPS)
        
        # Select Action
        if random.random() < epsilon:
            action_idx = random.randint(0, 2)
        else:
            with torch.no_grad():
                s = torch.tensor(state_img, dtype=torch.float).unsqueeze(0).to(device)
                sc = torch.tensor(scalar_vec, dtype=torch.float).unsqueeze(0).to(device)
                action_idx = policy_net(s, sc).argmax().item()
        
        # Step
        reward, term, trunc, score = game.step(action_idx)
        next_state_img, next_scalar_vec = game.get_state_bundle()
        
        # Store
        memory.add(state_img, scalar_vec, action_idx, reward, 
                   next_state_img, next_scalar_vec, term)
        
        game.render()

        # Optimize
        if len(memory) > 5000:
            beta = BETA_START + (1.0 - BETA_START) * min(1.0, steps_done / BETA_FRAMES)
            samples, indices, weights = memory.sample(BATCH_SIZE, beta)
            
            # Unpack...
            states, scalars, actions, rewards, next_states, next_scalars, dones = zip(*samples)
            
            # Conversion to tensors...
            states_t = torch.tensor(np.array(states), dtype=torch.float).to(device)
            scalars_t = torch.tensor(np.array(scalars), dtype=torch.float).to(device)
            actions_t = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
            rewards_t = torch.tensor(rewards, dtype=torch.float).unsqueeze(1).to(device)
            next_states_t = torch.tensor(np.array(next_states), dtype=torch.float).to(device)
            next_scalars_t = torch.tensor(np.array(next_scalars), dtype=torch.float).to(device)
            dones_t = torch.tensor(dones, dtype=torch.float).unsqueeze(1).to(device)
            weights_t = torch.tensor(weights, dtype=torch.float).unsqueeze(1).to(device)

            # Double DQN Calc...
            with torch.no_grad():
                next_actions = policy_net(next_states_t, next_scalars_t).argmax(1).unsqueeze(1)
                next_q_target = target_net(next_states_t, next_scalars_t).gather(1, next_actions)
            
            target_q = rewards_t + (GAMMA * next_q_target * (1 - dones_t))
            current_q = policy_net(states_t, scalars_t).gather(1, actions_t)
            
            td_errors = target_q - current_q
            loss = (weights_t * F.smooth_l1_loss(current_q, target_q, reduction='none')).mean()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
            optimizer.step()
            
            memory.update_priorities(indices, td_errors.detach().cpu().numpy().flatten())
            
            if steps_done % TARGET_UPDATE_INTERVAL == 0:
                target_net.load_state_dict(policy_net.state_dict())

        # Reset & Logging
        if term or trunc:
            game.reset()
            recent_scores.append(score)
            
        # Periodic Evaluation & Saving
        if steps_done > 0 and steps_done % EVAL_INTERVAL == 0:
            avg_train_score = sum(recent_scores)/len(recent_scores) if recent_scores else 0
            
            # Run specific Eval games (Epsilon=0)
            eval_score = evaluate(policy_net)
            
            print(f"Step {steps_done} | Train Avg: {avg_train_score:.1f} | Eval Score: {eval_score:.1f} | Eps: {epsilon:.2f}")
            
            # Save Best Model
            if eval_score > best_eval_score:
                best_eval_score = eval_score
                policy_net.save("model_best.pth")
                print(f"   >>> New Best Model Saved! ({best_eval_score:.1f})")

        if steps_done > 0 and steps_done % CHECKPOINT_INTERVAL == 0:
            policy_net.save("model_latest.pth")

        steps_done += 1

if __name__ == '__main__':
    train()