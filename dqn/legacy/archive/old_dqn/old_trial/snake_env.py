import random
import time

import numpy as np

# Absolute Directions
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3

# Relative Actions
ACTION_STRAIGHT, ACTION_RIGHT, ACTION_LEFT = 0, 1, 2
ACTION_SET = {ACTION_STRAIGHT, ACTION_RIGHT, ACTION_LEFT}


class SnakeEnv:
    def __init__(self, width=10, height=10, *, max_steps=None,
                 step_cost=0.0, allow_reverse=False, seed=None,
                 dist_reward_coef=0.0,
                 food_reward: float = 10.0,
                 death_penalty: float = -10.0):
        """
        Simple Snake environment.

        Args:
            width, height: grid dimensions.
            max_steps: optional cap on steps per episode (defaults to width*height*4).
            step_cost: per-step reward added each step (use small negative to discourage idling).
            allow_reverse: if False, ignore immediate 180° turns when snake length > 1.
            dist_reward_coef: small shaping reward toward food based on Manhattan distance.
            food_reward: reward for eating food.
            death_penalty: penalty for dying.
            seed: optional seed for reproducibility.
        """
        self.width = width
        self.height = height
        self.max_steps = max_steps or (width * height * 50)
        self.step_cost = step_cost
        self.allow_reverse = allow_reverse
        self.dist_reward_coef = dist_reward_coef
        self.food_reward = food_reward
        self.death_penalty = death_penalty
        self.seed(seed)
        self.reset()

    def reset(self, *, seed=None):
        if seed is not None:
            self.seed(seed)
        # center the snake
        cx = self.width // 2
        cy = self.height // 2
        self.snake = [(cx, cy)]  # head is index 0
        self.direction = RIGHT
        self.done = False
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0

        self._place_food()
        return self._get_obs()

    def seed(self, seed=None):
        self._seed = seed
        self.rng = random.Random(seed)
        np.random.seed(seed)
        return seed

    def _place_food(self):
        while True:
            pos = (self.rng.randrange(self.width),
                   self.rng.randrange(self.height))
            if pos not in self.snake:
                self.food = pos
                break

    def get_danger_features(self):
        """
        Returns a 3-element float32 array:
        [danger_straight, danger_left, danger_right] relative to current direction.
        1.0 if the cell in that relative direction is wall or snake body, else 0.0.
        """
        hx, hy = self.snake[0]

        if self.direction == UP:
            offsets = [(0, -1), (-1, 0), (1, 0)]  # straight, left, right
        elif self.direction == RIGHT:
            offsets = [(1, 0), (0, -1), (0, 1)]
        elif self.direction == DOWN:
            offsets = [(0, 1), (1, 0), (-1, 0)]
        else:  # LEFT
            offsets = [(-1, 0), (0, 1), (0, -1)]

        feats = []
        for dx, dy in offsets:
            nx, ny = hx + dx, hy + dy
            danger = (
                nx < 0 or nx >= self.width or
                ny < 0 or ny >= self.height or
                (nx, ny) in self.snake
            )
            feats.append(1.0 if danger else 0.0)

        return np.array(feats, dtype=np.float32)

    def step(self, action):
        if self.done:
            raise ValueError("Game is over. Call reset() before step().")

        # update direction from relative action
        if action is not None:
            if action not in ACTION_SET:
                raise ValueError(f"Invalid action {action}; must be one of {sorted(ACTION_SET)}")
            
            # 0=Straight, 1=Right, 2=Left
            if action == ACTION_RIGHT:
                self.direction = (self.direction + 1) % 4
            elif action == ACTION_LEFT:
                self.direction = (self.direction - 1) % 4
            # if ACTION_STRAIGHT, direction unchanged

        head_x, head_y = self.snake[0]

        if self.direction == UP:
            head_y -= 1
        elif self.direction == DOWN:
            head_y += 1
        elif self.direction == LEFT:
            head_x -= 1
        elif self.direction == RIGHT:
            head_x += 1

        new_head = (head_x, head_y)
        fx, fy = self.food
        old_dist = abs(head_x - fx) + abs(head_y - fy)
        reward = self.step_cost

        # hit wall or itself
        if (head_x < 0 or head_x >= self.width or
                head_y < 0 or head_y >= self.height or
                new_head in self.snake):
            self.done = True
            reward = self.death_penalty + self.step_cost
            
            reason = "Wall"
            if new_head in self.snake:
                reason = "Self"
            elif head_x < 0 or head_x >= self.width or head_y < 0 or head_y >= self.height:
                reason = "Wall"
                
            return self._get_obs(), reward, self.done, {"score": self.score, "death_reason": reason}

        # move snake
        self.snake.insert(0, new_head)

        # check food
        if new_head == self.food:
            self.score += 1
            reward = self.food_reward + self.step_cost
            self._place_food()
            self.steps_since_food = 0  # Reset hunger
        else:
            # remove tail
            self.snake.pop()
            self.steps_since_food += 1

        # Hunger check
        if self.steps_since_food >= 100:
            self.done = True
            return self._get_obs(), self.death_penalty, self.done, {"score": self.score, "death_reason": "Starvation"}

        if self.dist_reward_coef:
            # Encourage moving closer to food (dense shaping)
            new_dist = abs(new_head[0] - fx) + abs(new_head[1] - fy)
            reward += self.dist_reward_coef * (old_dist - new_dist)

        # episode cap (safety net)
        self.steps += 1
        if self.steps >= self.max_steps:
            self.done = True
            return self._get_obs(), reward, self.done, {"score": self.score, "death_reason": "Timeout"}

        return self._get_obs(), reward, self.done, {"score": self.score}
    def _get_obs(self):
        # Gradient CNN State: (3, H, W)
        # Channel 0: Head (1.0)
        # Channel 1: Body (Gradient 1.0 -> 0.0)
        # Channel 2: Food (1.0)
        
        grid = np.zeros((3, self.height, self.width), dtype=np.float32)

        # Channel 0: Head
        hx, hy = self.snake[0]
        grid[0, hy, hx] = 1.0

        # Channel 1: Body (Gradient encoding)
        # i=0 is the neck, i=len-1 is the tail.
        # We want the neck to be close to 1.0 (hot) and tail to be small (cool)
        snake_len = len(self.snake) - 1
        if snake_len > 0:
            for i, (x, y) in enumerate(self.snake[1:]):
                # Normalized value: Neck=1.0, Tail -> approaches 0
                grid[1, y, x] = 1.0 - (i / snake_len)

        # Channel 2: Food
        fx, fy = self.food
        grid[2, fy, fx] = 1.0

        return grid

    def render(self):
        """Simple ASCII render in the terminal."""
        grid = [["." for _ in range(self.width)] for _ in range(self.height)]

        fx, fy = self.food
        grid[fy][fx] = "F"

        for (x, y) in self.snake[1:]:
            grid[y][x] = "o"

        hx, hy = self.snake[0]
        grid[hy][hx] = "O"

        print("\n".join("".join(row) for row in grid))
        print(f"Score: {self.score}\n")

if __name__ == "__main__":
    import random
    import time

    env = SnakeEnv(width=10, height=10)
    obs = env.reset()
    done = False

    for step in range(50):  # you can lower this if you want fewer prints
        env.render()
        action = random.choice([0, 1, 2])  # Straight, Right, Left
        obs, reward, done, info = env.step(action)
        print(f"Reward: {reward} Score: {env.score}\n")
        time.sleep(0.1)

        if done:
            break
