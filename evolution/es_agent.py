"""
Evolution Strategy Agent for Snake.

Uses a small MLP with handcrafted features instead of CNN.
Weights are evolved via mutation, not trained via backprop.
"""

import numpy as np
from typing import List, Tuple, Optional
from snake_env import SnakeEnv, DIR_VECS, UP, RIGHT, DOWN, LEFT

# Relative directions for danger/food sensing
TURN_LEFT = -1
TURN_RIGHT = 1
STRAIGHT = 0


def get_features(env: SnakeEnv) -> np.ndarray:
    """
    Extract 16 simple O(1) features from current game state.
    No pathfinding or flood-fill - the agent must learn spatial reasoning.
    
    Features (16 total):
    - 3 danger sensors (straight, left, right) - binary
    - 4 food direction relative to snake heading
    - 1 food distance (Manhattan, normalized)
    - 4 wall distances (forward, left, right, behind) - normalized
    - 3 body distances in look directions - normalized
    - 1 length ratio
    """
    head = env.snake[0]
    hx, hy = head
    direction = env.direction
    body_set = set(env.snake[1:])
    
    # Direction vectors for relative moves
    def get_next_pos(turn: int) -> Tuple[int, int]:
        new_dir = (direction + turn) % 4
        dx, dy = DIR_VECS[new_dir]
        return (hx + dx, hy + dy)
    
    # Helper to check if position is dangerous
    def is_danger(pos: Tuple[int, int]) -> bool:
        x, y = pos
        if x < 0 or x >= env.width or y < 0 or y >= env.height:
            return True
        if pos in body_set and pos != env.snake[-1]:
            return True
        return False
    
    # 1. Danger sensors (3 features)
    danger_straight = float(is_danger(get_next_pos(STRAIGHT)))
    danger_left = float(is_danger(get_next_pos(TURN_LEFT)))
    danger_right = float(is_danger(get_next_pos(TURN_RIGHT)))
    
    # 2. Food direction relative to snake's heading (4 features)
    fx, fy = env.food
    food_dx = fx - hx
    food_dy = fy - hy
    
    forward_dx, forward_dy = DIR_VECS[direction]
    right_dx, right_dy = DIR_VECS[(direction + 1) % 4]
    
    forward_dot = food_dx * forward_dx + food_dy * forward_dy
    right_dot = food_dx * right_dx + food_dy * right_dy
    
    food_ahead = float(forward_dot > 0)
    food_behind = float(forward_dot < 0)
    food_right = float(right_dot > 0)
    food_left = float(right_dot < 0)
    
    # 3. Food distance (1 feature)
    food_dist = (abs(food_dx) + abs(food_dy)) / (env.width + env.height)
    
    # 4. Wall distances in each direction (4 features)
    def wall_distance(dx: int, dy: int) -> float:
        dist = 0
        x, y = hx, hy
        while True:
            x, y = x + dx, y + dy
            if x < 0 or x >= env.width or y < 0 or y >= env.height:
                break
            dist += 1
        return dist / max(env.width, env.height)
    
    wall_forward = wall_distance(*DIR_VECS[direction])
    wall_right = wall_distance(*DIR_VECS[(direction + 1) % 4])
    wall_left = wall_distance(*DIR_VECS[(direction - 1) % 4])
    wall_behind = wall_distance(*DIR_VECS[(direction + 2) % 4])
    
    # 5. Body distances - steps until hitting own body (3 features)
    def body_distance(dx: int, dy: int) -> float:
        dist = 0
        x, y = hx, hy
        max_dist = max(env.width, env.height)
        while dist < max_dist:
            x, y = x + dx, y + dy
            if x < 0 or x >= env.width or y < 0 or y >= env.height:
                return 1.0
            if (x, y) in body_set and (x, y) != env.snake[-1]:
                return dist / max_dist
            dist += 1
        return 1.0
    
    body_forward = body_distance(*DIR_VECS[direction])
    body_right = body_distance(*DIR_VECS[(direction + 1) % 4])
    body_left = body_distance(*DIR_VECS[(direction - 1) % 4])
    
    # 6. Length ratio (1 feature)
    length_ratio = len(env.snake) / (env.width * env.height)
    
    return np.array([
        danger_straight, danger_left, danger_right,
        food_ahead, food_right, food_behind, food_left,
        food_dist,
        wall_forward, wall_right, wall_left, wall_behind,
        body_forward, body_right, body_left,
        length_ratio
    ], dtype=np.float32)


class ESAgent:
    """
    Small MLP agent with evolvable weights.
    Architecture: 16 -> 32 -> 32 -> 3
    """
    
    def __init__(self, hidden1: int = 32, hidden2: int = 32):
        self.input_size = 16
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.output_size = 3  # straight, right, left
        
        # Initialize weights
        self.w1 = np.random.randn(self.input_size, hidden1).astype(np.float32) * 0.5
        self.b1 = np.zeros(hidden1, dtype=np.float32)
        self.w2 = np.random.randn(hidden1, hidden2).astype(np.float32) * 0.5
        self.b2 = np.zeros(hidden2, dtype=np.float32)
        self.w3 = np.random.randn(hidden2, self.output_size).astype(np.float32) * 0.5
        self.b3 = np.zeros(self.output_size, dtype=np.float32)
    
    def get_weights(self) -> np.ndarray:
        """Flatten all weights into a single vector."""
        return np.concatenate([
            self.w1.flatten(), self.b1,
            self.w2.flatten(), self.b2,
            self.w3.flatten(), self.b3
        ])
    
    def set_weights(self, weights: np.ndarray) -> None:
        """Set weights from a flattened vector."""
        idx = 0
        
        size = self.input_size * self.hidden1
        self.w1 = weights[idx:idx+size].reshape(self.input_size, self.hidden1)
        idx += size
        
        self.b1 = weights[idx:idx+self.hidden1]
        idx += self.hidden1
        
        size = self.hidden1 * self.hidden2
        self.w2 = weights[idx:idx+size].reshape(self.hidden1, self.hidden2)
        idx += size
        
        self.b2 = weights[idx:idx+self.hidden2]
        idx += self.hidden2
        
        size = self.hidden2 * self.output_size
        self.w3 = weights[idx:idx+size].reshape(self.hidden2, self.output_size)
        idx += size
        
        self.b3 = weights[idx:idx+self.output_size]
    
    def num_params(self) -> int:
        """Total number of parameters."""
        return len(self.get_weights())
    
    def forward(self, features: np.ndarray) -> np.ndarray:
        """Forward pass, returns action probabilities."""
        # ReLU activations
        x = np.maximum(0, features @ self.w1 + self.b1)
        x = np.maximum(0, x @ self.w2 + self.b2)
        logits = x @ self.w3 + self.b3
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / exp_logits.sum()
    
    def act(self, env: SnakeEnv, deterministic: bool = True) -> int:
        """Choose action given current environment state."""
        features = get_features(env)
        probs = self.forward(features)
        
        if deterministic:
            return int(np.argmax(probs))
        else:
            return int(np.random.choice(3, p=probs))
    
    def copy(self) -> 'ESAgent':
        """Create a deep copy of this agent."""
        new_agent = ESAgent(self.hidden1, self.hidden2)
        new_agent.set_weights(self.get_weights().copy())
        return new_agent
    
    def mutate(self, sigma: float = 0.1) -> 'ESAgent':
        """Create a mutated copy of this agent."""
        new_agent = self.copy()
        weights = new_agent.get_weights()
        weights += np.random.randn(len(weights)).astype(np.float32) * sigma
        new_agent.set_weights(weights)
        return new_agent
    
    def save(self, path: str) -> None:
        """Save weights to file."""
        np.savez(path, 
                 w1=self.w1, b1=self.b1,
                 w2=self.w2, b2=self.b2,
                 w3=self.w3, b3=self.b3,
                 hidden1=self.hidden1, hidden2=self.hidden2)
    
    @classmethod
    def load(cls, path: str) -> 'ESAgent':
        """Load agent from file."""
        data = np.load(path)
        agent = cls(int(data['hidden1']), int(data['hidden2']))
        agent.w1 = data['w1']
        agent.b1 = data['b1']
        agent.w2 = data['w2']
        agent.b2 = data['b2']
        agent.w3 = data['w3']
        agent.b3 = data['b3']
        return agent


def evaluate_agent(agent: ESAgent, num_games: int = 5, 
                   max_steps: int = 2000, seed: Optional[int] = None) -> Tuple[float, float, float, float, float]:
    """
    Evaluate an agent over multiple games.
    
    Returns: (avg_score, avg_length, avg_coverage, max_score, max_coverage)
    """
    scores = []
    lengths = []
    coverages = []
    
    for i in range(num_games):
        game_seed = None if seed is None else seed + i
        env = SnakeEnv(width=10, height=10, seed=game_seed, 
                       start_length_min=3, start_length_max=3)
        
        steps = 0
        while not env.done and steps < max_steps:
            action = agent.act(env, deterministic=True)
            env.step(action)
            steps += 1
        
        scores.append(env.score)
        lengths.append(len(env.snake))
        coverages.append(len(env.snake) / (env.width * env.height))
    
    return (np.mean(scores), np.mean(lengths), np.mean(coverages), 
            max(scores), max(coverages))
