import argparse
import sys
from typing import Optional
from pathlib import Path

import torch

# Add parent directory to path to find snake_env
sys.path.append(str(Path(__file__).parent.parent))

from dqn_model import DuelingDQN
from snake_env import SnakeEnv

try:
    from train_dqn import ENV_CONFIG
except Exception:
    ENV_CONFIG = {
        "width": 10,
        "height": 10,
        "obs_mode": "gradient",
        "start_length_min": 1,
        "start_length_max": 15,
        "step_cost": -0.05,
        "food_reward": 25.0,
        "death_penalty": -10.0,
        "timeout_factor": 50,
        "timeout_penalty": -10.0,
        "grow_on_food": True,
        "growth_interval": 0,
        "growth_max_length": None,
        "stall_penalty_coef": 0.05,
        "loop_penalty_coef": 0.2,
        "loop_window_base": 0,
        "loop_window_per_len": 2,
        "body_penalty_coef": 0.05,
        "body_penalty_power": 2.0,
        "tail_reward_coef": 0.0,
        "exclude_tail_from_penalty": True,
        "render_growth_flash_frames": 0,
    }


def load_model(
    checkpoint_path: str,
    device: torch.device,
    observation_shape,
    num_actions: int = 3,
) -> DuelingDQN:
    model = DuelingDQN(observation_shape, num_actions=num_actions, scalar_dim=3).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "policy_state" in checkpoint:
        model.load_state_dict(checkpoint["policy_state"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def make_env(
    seed: Optional[int],
    fps: int,
    cell_size: int,
    timeout_factor: int,
    timeout_penalty: float,
    stall_penalty_coef: float,
    loop_penalty_coef: float,
    loop_window_base: int,
    loop_window_per_len: int,
    obs_mode: str,
) -> SnakeEnv:
    return SnakeEnv(
        width=ENV_CONFIG["width"],
        height=ENV_CONFIG["height"],
        obs_mode=obs_mode,
        start_length_min=ENV_CONFIG["start_length_min"],
        start_length_max=ENV_CONFIG["start_length_max"],
        step_cost=ENV_CONFIG["step_cost"],
        food_reward=ENV_CONFIG["food_reward"],
        death_penalty=ENV_CONFIG["death_penalty"],
        timeout_penalty=timeout_penalty,
        max_steps_without_food_factor=timeout_factor,
        grow_on_food=ENV_CONFIG["grow_on_food"],
        growth_interval=ENV_CONFIG["growth_interval"],
        growth_max_length=ENV_CONFIG["growth_max_length"],
        stall_penalty_coef=stall_penalty_coef,
        loop_penalty_coef=loop_penalty_coef,
        loop_window_base=loop_window_base,
        loop_window_per_len=loop_window_per_len,
        body_penalty_coef=ENV_CONFIG["body_penalty_coef"],
        body_penalty_power=ENV_CONFIG["body_penalty_power"],
        tail_reward_coef=ENV_CONFIG["tail_reward_coef"],
        exclude_tail_from_penalty=ENV_CONFIG["exclude_tail_from_penalty"],
        render_growth_flash_frames=ENV_CONFIG["render_growth_flash_frames"],
        render_mode="human",
        render_fps=fps,
        render_cell_size=cell_size,
        seed=seed,
    )


def run(
    checkpoint_path: str,
    episodes: int,
    fps: int,
    cell_size: int,
    timeout_factor: int,
    timeout_penalty: float,
    stall_penalty_coef: float,
    loop_penalty_coef: float,
    loop_window_base: int,
    loop_window_per_len: int,
    obs_mode: str,
    start_length: Optional[int],
    seed: Optional[int],
    debug_every: int,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(
        seed=seed,
        fps=fps,
        cell_size=cell_size,
        timeout_factor=timeout_factor,
        timeout_penalty=timeout_penalty,
        stall_penalty_coef=stall_penalty_coef,
        loop_penalty_coef=loop_penalty_coef,
        loop_window_base=loop_window_base,
        loop_window_per_len=loop_window_per_len,
        obs_mode=obs_mode,
    )
    model = load_model(checkpoint_path, device, env.observation_shape)

    for ep in range(1, episodes + 1):
        state, scalars = env.reset(start_length=start_length)
        done = False
        step_idx = 0
        while not done:
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                sc = torch.tensor(scalars, dtype=torch.float32, device=device).unsqueeze(0)
                action = int(model(s, sc).argmax(dim=1).item())

            state, scalars, reward, done, info = env.step(action)
            step_idx += 1
            if debug_every and step_idx % debug_every == 0:
                head = env.snake[0]
                food = env.food
                dist = abs(head[0] - food[0]) + abs(head[1] - food[1])
                print(
                    "debug "
                    f"step={step_idx} action={action} reward={reward:.3f} "
                    f"len={len(env.snake)} dist={dist} steps_since_food={env.steps_since_food} "
                    f"scalars={scalars.tolist()}"
                )
            env.render()

        length = info.get("length", len(env.snake))
        print(
            f"Episode {ep} | Score: {info['score']} | Length: {length} | "
            f"Death: {info.get('death_reason')}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch a trained Snake agent")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="dqn/checkpoints/model_best.pth",
        help="Path to model_best.pth or a checkpoint_*.pth file",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--cell-size", type=int, default=40)
    parser.add_argument("--timeout-factor", type=int, default=ENV_CONFIG["timeout_factor"])
    parser.add_argument("--timeout-penalty", type=float, default=ENV_CONFIG["timeout_penalty"])
    parser.add_argument(
        "--stall-penalty-coef", type=float, default=ENV_CONFIG["stall_penalty_coef"]
    )
    parser.add_argument(
        "--loop-penalty-coef", type=float, default=ENV_CONFIG["loop_penalty_coef"]
    )
    parser.add_argument(
        "--loop-window-base", type=int, default=ENV_CONFIG["loop_window_base"]
    )
    parser.add_argument(
        "--loop-window-per-len", type=int, default=ENV_CONFIG["loop_window_per_len"]
    )
    parser.add_argument(
        "--obs-mode",
        type=str,
        default=ENV_CONFIG["obs_mode"],
        choices=["gradient", "tail"],
    )
    parser.add_argument("--start-length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--debug-every", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.checkpoint,
        args.episodes,
        args.fps,
        args.cell_size,
        args.timeout_factor,
        args.timeout_penalty,
        args.stall_penalty_coef,
        args.loop_penalty_coef,
        args.loop_window_base,
        args.loop_window_per_len,
        args.obs_mode,
        args.start_length,
        args.seed,
        args.debug_every,
    )
