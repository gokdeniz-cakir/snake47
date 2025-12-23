import argparse
import os
import time
from typing import Optional

import torch

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


def load_checkpoint_into_model(
    model: DuelingDQN, checkpoint_path: str, device: torch.device
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "policy_state" in checkpoint:
        state = checkpoint["policy_state"]
    else:
        state = checkpoint
    model.load_state_dict(state)
    model.eval()


def try_load(
    model: DuelingDQN,
    checkpoint_path: str,
    device: torch.device,
    retries: int,
    retry_delay: float,
) -> bool:
    for _ in range(retries):
        try:
            load_checkpoint_into_model(model, checkpoint_path, device)
            return True
        except Exception:
            time.sleep(retry_delay)
    return False


def run(
    save_dir: str,
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
    poll_seconds: float,
    retries: int,
    retry_delay: float,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(
        seed,
        fps,
        cell_size,
        timeout_factor,
        timeout_penalty,
        stall_penalty_coef,
        loop_penalty_coef,
        loop_window_base,
        loop_window_per_len,
        obs_mode,
    )
    model = DuelingDQN(env.observation_shape, num_actions=3, scalar_dim=3).to(device)

    checkpoint_path = os.path.join(save_dir, "checkpoint_latest.pth")
    last_mtime: Optional[float] = None
    waiting = False

    while True:
        if not os.path.exists(checkpoint_path):
            if not waiting:
                print(f"Waiting for checkpoint: {checkpoint_path}")
                waiting = True
            time.sleep(poll_seconds)
            continue

        mtime = os.path.getmtime(checkpoint_path)
        if last_mtime is None or mtime > last_mtime:
            if try_load(model, checkpoint_path, device, retries, retry_delay):
                last_mtime = mtime
                waiting = False
                print(f"Loaded checkpoint: {checkpoint_path}")
            else:
                time.sleep(poll_seconds)
                continue

        for ep in range(1, episodes + 1):
            state, scalars = env.reset(start_length=start_length)
            done = False
            while not done:
                with torch.no_grad():
                    s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    sc = torch.tensor(scalars, dtype=torch.float32, device=device).unsqueeze(0)
                    action = int(model(s, sc).argmax(dim=1).item())

                state, scalars, _, done, info = env.step(action)
                env.render()

            length = info.get("length", len(env.snake))
            print(
                f"Episode {ep} | Score: {info['score']} | Length: {length} | "
                f"Death: {info.get('death_reason')}"
            )

            if os.path.exists(checkpoint_path):
                latest_mtime = os.path.getmtime(checkpoint_path)
                if latest_mtime > last_mtime:
                    break

        time.sleep(poll_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live watch the latest checkpoint")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
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
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry-delay", type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        save_dir=args.save_dir,
        episodes=args.episodes,
        fps=args.fps,
        cell_size=args.cell_size,
        timeout_factor=args.timeout_factor,
        timeout_penalty=args.timeout_penalty,
        stall_penalty_coef=args.stall_penalty_coef,
        loop_penalty_coef=args.loop_penalty_coef,
        loop_window_base=args.loop_window_base,
        loop_window_per_len=args.loop_window_per_len,
        obs_mode=args.obs_mode,
        start_length=args.start_length,
        seed=args.seed,
        poll_seconds=args.poll_seconds,
        retries=args.retries,
        retry_delay=args.retry_delay,
    )
