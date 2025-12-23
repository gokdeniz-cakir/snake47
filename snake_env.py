import random
from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3

ACTION_STRAIGHT = 0
ACTION_RIGHT = 1
ACTION_LEFT = 2
ACTION_SET = (ACTION_STRAIGHT, ACTION_RIGHT, ACTION_LEFT)

DIR_VECS = {
    UP: (0, -1),
    RIGHT: (1, 0),
    DOWN: (0, 1),
    LEFT: (-1, 0),
}


class SnakeEnv:
    def __init__(
        self,
        width: int = 10,
        height: int = 10,
        *,
        obs_mode: str = "gradient",
        start_length_min: int = 3,
        start_length_max: int = 3,
        step_cost: float = -0.01,
        food_reward: float = 10.0,
        death_penalty: float = -10.0,
        timeout_penalty: float = -1.0,
        max_steps_without_food_factor: int = 100,
        grow_on_food: bool = True,
        growth_interval: int = 0,
        growth_max_length: Optional[int] = None,
        stall_penalty_coef: float = 0.05,
        loop_penalty_coef: float = 0.0,
        loop_window_base: int = 0,
        loop_window_per_len: int = 0,
        body_penalty_coef: float = 0.05,
        body_penalty_power: float = 2.0,
        tail_reward_coef: float = 0.0,
        exclude_tail_from_penalty: bool = True,
        render_mode: Optional[str] = None,
        render_fps: int = 30,
        render_cell_size: int = 20,
        render_growth_flash_frames: int = 0,
        seed: Optional[int] = None,
    ):
        self.width = width
        self.height = height
        self.obs_mode = obs_mode
        self.start_length_min = max(1, int(start_length_min))
        self.start_length_max = int(start_length_max)
        if self.start_length_max < self.start_length_min:
            raise ValueError("start_length_max must be >= start_length_min")
        self.step_cost = step_cost
        self.food_reward = food_reward
        self.death_penalty = death_penalty
        self.timeout_penalty = timeout_penalty
        self.max_steps_without_food_factor = max_steps_without_food_factor
        self.grow_on_food = bool(grow_on_food)
        self.growth_interval = max(0, int(growth_interval))
        if growth_max_length is None:
            self.growth_max_length = self.width * self.height
        else:
            self.growth_max_length = max(
                1, min(int(growth_max_length), self.width * self.height)
            )
        self.stall_penalty_coef = stall_penalty_coef
        self.loop_penalty_coef = loop_penalty_coef
        self.loop_window_base = max(0, int(loop_window_base))
        self.loop_window_per_len = max(0, int(loop_window_per_len))
        self.body_penalty_coef = body_penalty_coef
        self.body_penalty_power = body_penalty_power
        self.tail_reward_coef = tail_reward_coef
        self.exclude_tail_from_penalty = exclude_tail_from_penalty
        self.render_mode = render_mode
        self.render_fps = render_fps
        self.render_cell_size = render_cell_size
        self.render_growth_flash_frames = max(0, int(render_growth_flash_frames))
        self._pygame = None
        self._screen = None
        self._clock = None
        self.seed(seed)
        self.reset()

    @property
    def observation_shape(self) -> Tuple[int, int, int]:
        if self.obs_mode == "gradient":
            channels = 4
        elif self.obs_mode == "tail":
            channels = 5
        else:
            raise ValueError(f"Unknown obs_mode: {self.obs_mode!r}")
        return (channels, self.height + 2, self.width + 2)

    def seed(self, seed: Optional[int] = None) -> Optional[int]:
        self._seed = seed
        self._rng = random.Random(seed)
        np.random.seed(seed)
        return seed

    def reset(self, start_length: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        length = self._sample_start_length() if start_length is None else int(start_length)
        length = max(1, min(length, self.width * self.height))

        if (
            start_length is None
            and self.start_length_min == 3
            and self.start_length_max == 3
            and length == 3
        ):
            cx = self.width // 2
            cy = self.height // 2
            self.snake = [(cx, cy), (cx - 1, cy), (cx - 2, cy)]
            self.direction = RIGHT
        else:
            self.snake, self.direction = self._generate_snake(length)

        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.done = False
        self._loop_states = deque()
        self._loop_counts = {}
        self._flash_tail_steps = 0
        self._place_food()
        return self._get_obs(), self._get_scalars()

    def _sample_start_length(self) -> int:
        return self._rng.randint(self.start_length_min, self.start_length_max)

    def _count_safe_moves(self, snake, direction: int) -> int:
        head_x, head_y = snake[0]
        safe = 0

        for action in ACTION_SET:
            next_dir = direction
            if action == ACTION_RIGHT:
                next_dir = (direction + 1) % 4
            elif action == ACTION_LEFT:
                next_dir = (direction - 1) % 4

            dx, dy = DIR_VECS[next_dir]
            nx, ny = head_x + dx, head_y + dy
            if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
                continue

            body_to_check = snake[1:-1]
            if (nx, ny) in body_to_check:
                continue
            safe += 1

        return safe

    def _generate_snake(self, length: int):
        if length == 1:
            head = (self._rng.randrange(self.width), self._rng.randrange(self.height))
            direction = self._rng.choice([UP, RIGHT, DOWN, LEFT])
            return [head], direction

        dir_from_vec = {v: k for k, v in DIR_VECS.items()}

        def initial_direction(snake):
            hx, hy = snake[0]
            nx, ny = snake[1]
            vec = (hx - nx, hy - ny)
            return dir_from_vec.get(vec)

        best = None
        best_safe = -1

        for _ in range(500):
            head = (self._rng.randrange(self.width), self._rng.randrange(self.height))
            path = [head]
            while len(path) < length:
                x, y = path[-1]
                candidates = []
                for dx, dy in DIR_VECS.values():
                    nx, ny = x + dx, y + dy
                    if (
                        0 <= nx < self.width
                        and 0 <= ny < self.height
                        and (nx, ny) not in path
                    ):
                        candidates.append((nx, ny))
                if not candidates:
                    break
                path.append(self._rng.choice(candidates))

            if len(path) != length:
                continue

            for snake in (path, list(reversed(path))):
                direction = initial_direction(snake)
                if direction is None:
                    continue
                safe = self._count_safe_moves(snake, direction)
                if safe > best_safe:
                    best_safe = safe
                    best = (snake, direction)

            if best is not None and best_safe > 0:
                return best

        cells = []
        for y in range(self.height):
            xs = range(self.width) if (y % 2 == 0) else range(self.width - 1, -1, -1)
            for x in xs:
                cells.append((x, y))
        start = self._rng.randrange(0, len(cells) - length + 1)
        path = cells[start : start + length]
        snake = path
        direction = initial_direction(snake)
        if direction is None:
            direction = RIGHT
        return snake, direction

    def _place_food(self) -> None:
        while True:
            pos = (self._rng.randrange(self.width), self._rng.randrange(self.height))
            if pos not in self.snake:
                self.food = pos
                break

    def _update_direction(self, action: int) -> None:
        if action == ACTION_RIGHT:
            self.direction = (self.direction + 1) % 4
        elif action == ACTION_LEFT:
            self.direction = (self.direction - 1) % 4

    def _body_potential(self) -> float:
        if len(self.snake) <= 2:
            return 0.0

        if self.exclude_tail_from_penalty:
            segments = self.snake[1:-1]
        else:
            segments = self.snake[1:]

        hx, hy = self.snake[0]
        total = 0.0
        for x, y in segments:
            dist = abs(hx - x) + abs(hy - y)
            if dist == 0:
                continue
            total += 1.0 / (dist ** self.body_penalty_power)
        return total

    def _tail_distance(self) -> int:
        if len(self.snake) <= 1:
            return 0
        hx, hy = self.snake[0]
        tx, ty = self.snake[-1]
        return abs(hx - tx) + abs(hy - ty)

    def step(self, action: int) -> Tuple[np.ndarray, np.ndarray, float, bool, Dict]:
        if self.done:
            raise ValueError("Episode is done. Call reset() before step().")

        body_potential_before = (
            self._body_potential() if self.body_penalty_coef != 0.0 else 0.0
        )
        tail_dist_before = self._tail_distance() if self.tail_reward_coef != 0.0 else 0

        self._update_direction(action)
        dx, dy = DIR_VECS[self.direction]
        next_head = (self.snake[0][0] + dx, self.snake[0][1] + dy)

        death_reason = None
        if (
            next_head[0] < 0
            or next_head[0] >= self.width
            or next_head[1] < 0
            or next_head[1] >= self.height
        ):
            self.done = True
            death_reason = "wall"
            reward = self.death_penalty
            info = {
                "score": self.score,
                "death_reason": death_reason,
                "length": len(self.snake),
            }
            return self._get_obs(), self._get_scalars(), reward, True, info

        next_step = self.steps + 1
        scheduled_grow = (
            self.growth_interval > 0
            and (next_step % self.growth_interval == 0)
            and len(self.snake) < self.growth_max_length
        )

        will_eat = next_head == self.food
        will_grow = (will_eat and self.grow_on_food) or scheduled_grow
        if will_grow:
            body_to_check = self.snake[1:]
        else:
            body_to_check = self.snake[1:-1]

        if next_head in body_to_check:
            self.done = True
            death_reason = "self"
            reward = self.death_penalty
            info = {
                "score": self.score,
                "death_reason": death_reason,
                "length": len(self.snake),
            }
            return self._get_obs(), self._get_scalars(), reward, True, info

        self.snake.insert(0, next_head)

        reward = self.step_cost
        if will_eat:
            self.score += 1
            if self.grow_on_food:
                reward += self.food_reward
            self._place_food()
            self.steps_since_food = 0
        else:
            self.steps_since_food += 1

        if not will_grow:
            self.snake.pop()
        elif self.render_growth_flash_frames > 0:
            self._flash_tail_steps = self.render_growth_flash_frames

        self.steps += 1
        max_steps_without_food = self.max_steps_without_food_factor * len(self.snake)
        if self.steps_since_food >= max_steps_without_food:
            self.done = True
            death_reason = "timeout"
            reward = self.timeout_penalty

        if not self.done:
            if self.stall_penalty_coef != 0.0 and max_steps_without_food > 0:
                stall_frac = self.steps_since_food / max_steps_without_food
                reward -= self.stall_penalty_coef * stall_frac
            if self.loop_penalty_coef != 0.0 and (
                self.loop_window_base > 0 or self.loop_window_per_len > 0
            ):
                if self.loop_window_per_len > 0:
                    window_size = self.loop_window_per_len * len(self.snake)
                else:
                    window_size = self.loop_window_base
                if window_size > 0:
                    state_key = (self.snake[0], self.direction, self.food)
                    if state_key in self._loop_counts:
                        reward -= self.loop_penalty_coef
                    self._loop_states.append(state_key)
                    self._loop_counts[state_key] = self._loop_counts.get(state_key, 0) + 1
                    while len(self._loop_states) > window_size:
                        old_key = self._loop_states.popleft()
                        self._loop_counts[old_key] -= 1
                        if self._loop_counts[old_key] <= 0:
                            del self._loop_counts[old_key]

            if self.body_penalty_coef != 0.0:
                body_potential_after = self._body_potential()
                reward += self.body_penalty_coef * (
                    body_potential_before - body_potential_after
                )

            if self.tail_reward_coef != 0.0:
                tail_dist_after = self._tail_distance()
                reward += self.tail_reward_coef * (tail_dist_before - tail_dist_after)

        info = {
            "score": self.score,
            "death_reason": death_reason,
            "ate_food": will_eat,
            "steps_since_food": self.steps_since_food,
            "length": len(self.snake),
        }
        return self._get_obs(), self._get_scalars(), reward, self.done, info

    def _get_obs(self) -> np.ndarray:
        if self.obs_mode == "gradient":
            obs = np.zeros((4, self.height + 2, self.width + 2), dtype=np.float32)
            walls_chan = 3
        elif self.obs_mode == "tail":
            obs = np.zeros((5, self.height + 2, self.width + 2), dtype=np.float32)
            walls_chan = 4
        else:
            raise ValueError(f"Unknown obs_mode: {self.obs_mode!r}")

        obs[walls_chan, 0, :] = 1.0
        obs[walls_chan, -1, :] = 1.0
        obs[walls_chan, :, 0] = 1.0
        obs[walls_chan, :, -1] = 1.0

        def to_padded(pos: Tuple[int, int]) -> Tuple[int, int]:
            return pos[1] + 1, pos[0] + 1

        head_r, head_c = to_padded(self.snake[0])
        obs[0, head_r, head_c] = 1.0

        food_r, food_c = to_padded(self.food)
        obs[1, food_r, food_c] = 1.0

        if self.obs_mode == "gradient":
            body_len = len(self.snake) - 1
            if body_len > 0:
                for i, (x, y) in enumerate(self.snake[1:]):
                    r, c = to_padded((x, y))
                    obs[2, r, c] = 1.0 - (i / body_len)
        else:
            if len(self.snake) > 2:
                for x, y in self.snake[1:-1]:
                    r, c = to_padded((x, y))
                    obs[2, r, c] = 1.0
            if len(self.snake) > 1:
                tx, ty = self.snake[-1]
                tr, tc = to_padded((tx, ty))
                obs[3, tr, tc] = 1.0

        return obs

    def _get_scalars(self) -> np.ndarray:
        dx, dy = DIR_VECS[self.direction]
        norm_len = len(self.snake) / (self.width * self.height)
        return np.array([norm_len, dx, dy], dtype=np.float32)

    def _init_render(self) -> None:
        import pygame

        pygame.init()
        self._pygame = pygame
        self._screen = pygame.display.set_mode(
            (self.width * self.render_cell_size, self.height * self.render_cell_size)
        )
        pygame.display.set_caption("Snake")
        self._clock = pygame.time.Clock()

    def render(self) -> None:
        if self.render_mode != "human":
            return

        if self._pygame is None:
            self._init_render()

        pygame = self._pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                raise SystemExit

        self._screen.fill((0, 0, 0))

        tail = self.snake[-1]
        for x, y in self.snake:
            rect = pygame.Rect(
                x * self.render_cell_size,
                y * self.render_cell_size,
                self.render_cell_size,
                self.render_cell_size,
            )
            color = (0, 200, 0)
            if self._flash_tail_steps > 0 and (x, y) == tail:
                color = (255, 255, 0)
            pygame.draw.rect(self._screen, color, rect)

        hx, hy = self.snake[0]
        head_rect = pygame.Rect(
            hx * self.render_cell_size,
            hy * self.render_cell_size,
            self.render_cell_size,
            self.render_cell_size,
        )
        pygame.draw.rect(self._screen, (0, 255, 255), head_rect)

        fx, fy = self.food
        food_rect = pygame.Rect(
            fx * self.render_cell_size,
            fy * self.render_cell_size,
            self.render_cell_size,
            self.render_cell_size,
        )
        pygame.draw.rect(self._screen, (255, 0, 0), food_rect)

        pygame.display.flip()
        self._clock.tick(self.render_fps)
        if self._flash_tail_steps > 0:
            self._flash_tail_steps -= 1

    def close(self) -> None:
        if self._pygame is not None:
            self._pygame.quit()
            self._pygame = None
            self._screen = None
            self._clock = None
