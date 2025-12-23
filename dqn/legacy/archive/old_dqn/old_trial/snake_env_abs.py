# To test an old checkpoint. not up to date file. similar case with play_abs.py
import random
import numpy as np

# Actions (Absolute)
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
ACTION_SET = {UP, RIGHT, DOWN, LEFT}
REVERSE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}

class SnakeEnv:
    def __init__(self, width=10, height=10, *, max_steps=None,
                 step_cost=0.0, allow_reverse=False, seed=None,
                 dist_reward_coef=0.0,
                 food_reward: float = 10.0,
                 death_penalty: float = -10.0):
        self.width = width
        self.height = height
        self.max_steps = max_steps if max_steps is not None else 100000000000000000000000000000000
        self.step_cost = step_cost
        self.allow_reverse = allow_reverse
        self.dist_reward_coef = dist_reward_coef
        self.food_reward = food_reward
        self.death_penalty = death_penalty
        self.seed = seed

        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            self.seed = seed
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        self.snake = [(self.width // 2, self.height // 2)]
        self.food = self._place_food()
        self.score = 0
        self.steps = 0
        self.done = False
        self.direction = RIGHT
        return self._get_obs()

    def _place_food(self):
        while True:
            fx = random.randint(0, self.width - 1)
            fy = random.randint(0, self.height - 1)
            if (fx, fy) not in self.snake:
                return (fx, fy)

    def step(self, action):
        if self.done:
            raise ValueError("Game is over. Call reset() before step().")

        if action is not None:
            if action not in ACTION_SET:
                raise ValueError(f"Invalid action {action}")
            
            # prevent instant 180° turns if configured
            if not self.allow_reverse and len(self.snake) > 1 and REVERSE[action] == self.direction:
                action = self.direction
            self.direction = int(action)

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

        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.score += 1
            reward = self.food_reward + self.step_cost
            self.food = self._place_food()
        else:
            self.snake.pop()

        if self.dist_reward_coef:
            new_dist = abs(new_head[0] - fx) + abs(new_head[1] - fy)
            reward += self.dist_reward_coef * (old_dist - new_dist)

        self.steps += 1
        if self.steps >= self.max_steps:
            self.done = True
            return self._get_obs(), reward, self.done, {"score": self.score, "death_reason": "Timeout"}

        return self._get_obs(), reward, self.done, {"score": self.score}

    def _get_obs(self):
        grid = np.zeros((3, self.height, self.width), dtype=np.float32)
        hx, hy = self.snake[0]
        grid[0, hy, hx] = 1.0
        for (x, y) in self.snake[1:]:
            grid[1, y, x] = 1.0
        fx, fy = self.food
        grid[2, fy, fx] = 1.0
        return grid

    def render(self):
        pass
