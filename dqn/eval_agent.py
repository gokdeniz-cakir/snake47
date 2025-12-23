import argparse
import sys
from collections import Counter
from typing import Optional
from pathlib import Path

import numpy as np
import torch

# Add parent directory to path to find snake_env
sys.path.append(str(Path(__file__).parent.parent))

from dqn_model import DuelingDQN
from snake_env import SnakeEnv

try:
    from train_dqn import ENV_CONFIG, EVAL_CONFIG
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
    EVAL_CONFIG = {
        "interval": 10_000,
        "episodes": 20,
        "seed": 123,
        "start_length_min": 3,
        "start_length_max": 3,
    }


def make_env(
    timeout_factor: int,
    timeout_penalty: float,
    stall_penalty_coef: float,
    loop_penalty_coef: float,
    loop_window_base: int,
    loop_window_per_len: int,
    seed: Optional[int],
    obs_mode: str,
    start_length_min: int,
    start_length_max: int,
) -> SnakeEnv:
    return SnakeEnv(
        width=ENV_CONFIG["width"],
        height=ENV_CONFIG["height"],
        obs_mode=obs_mode,
        start_length_min=start_length_min,
        start_length_max=start_length_max,
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
        render_mode=None,
        seed=seed,
    )


def load_model(checkpoint_path: str, env: SnakeEnv, device: torch.device) -> DuelingDQN:
    model = DuelingDQN(env.observation_shape, num_actions=3, scalar_dim=3).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "policy_state" in checkpoint:
        model.load_state_dict(checkpoint["policy_state"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def evaluate(
    model: DuelingDQN,
    device: torch.device,
    episodes: int,
    seed: Optional[int],
    start_length: Optional[int],
    timeout_factor: int,
    timeout_penalty: float,
    stall_penalty_coef: float,
    loop_penalty_coef: float,
    loop_window_base: int,
    loop_window_per_len: int,
    obs_mode: str,
) -> None:
    env = make_env(
        timeout_factor=timeout_factor,
        timeout_penalty=timeout_penalty,
        stall_penalty_coef=stall_penalty_coef,
        loop_penalty_coef=loop_penalty_coef,
        loop_window_base=loop_window_base,
        loop_window_per_len=loop_window_per_len,
        seed=seed,
        obs_mode=obs_mode,
        start_length_min=EVAL_CONFIG["start_length_min"],
        start_length_max=EVAL_CONFIG["start_length_max"],
    )
    scores = []
    returns = []
    lengths = []
    deaths = Counter()

    with torch.no_grad():
        for ep in range(episodes):
            if seed is not None:
                env.seed(seed + ep)
            state, scalars = env.reset(start_length=start_length)
            done = False
            total_reward = 0.0

            while not done:
                s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                sc = torch.tensor(scalars, dtype=torch.float32, device=device).unsqueeze(0)
                action = int(model(s, sc).argmax(dim=1).item())
                state, scalars, reward, done, info = env.step(action)
                total_reward += float(reward)

            scores.append(int(info["score"]))
            returns.append(total_reward)
            deaths[info.get("death_reason")] += 1
            lengths.append(int(info.get("length", len(env.snake))))

    scores_arr = np.array(scores, dtype=np.float32)
    returns_arr = np.array(returns, dtype=np.float32)
    lengths_arr = np.array(lengths, dtype=np.float32)

    def fmt(x: float) -> str:
        return f"{x:.2f}"

    print(f"Episodes: {episodes}")
    print(f"Score  mean/std/min/med/max: {fmt(scores_arr.mean())} / {fmt(scores_arr.std())} / "
          f"{fmt(scores_arr.min())} / {fmt(np.median(scores_arr))} / {fmt(scores_arr.max())}")
    print(f"Return mean/std/min/med/max: {fmt(returns_arr.mean())} / {fmt(returns_arr.std())} / "
          f"{fmt(returns_arr.min())} / {fmt(np.median(returns_arr))} / {fmt(returns_arr.max())}")
    print(f"Length mean/std/min/med/max: {fmt(lengths_arr.mean())} / {fmt(lengths_arr.std())} / "
          f"{fmt(lengths_arr.min())} / {fmt(np.median(lengths_arr))} / {fmt(lengths_arr.max())}")
    print(f"Death reasons: {dict(deaths)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Snake agent")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=EVAL_CONFIG["seed"])
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(
        timeout_factor=args.timeout_factor,
        timeout_penalty=args.timeout_penalty,
        stall_penalty_coef=args.stall_penalty_coef,
        loop_penalty_coef=args.loop_penalty_coef,
        loop_window_base=args.loop_window_base,
        loop_window_per_len=args.loop_window_per_len,
        seed=args.seed,
        obs_mode=args.obs_mode,
        start_length_min=EVAL_CONFIG["start_length_min"],
        start_length_max=EVAL_CONFIG["start_length_max"],
    )
    model = load_model(args.checkpoint, env, device)
    evaluate(
        model,
        device,
        episodes=args.episodes,
        seed=args.seed,
        start_length=args.start_length,
        timeout_factor=args.timeout_factor,
        timeout_penalty=args.timeout_penalty,
        stall_penalty_coef=args.stall_penalty_coef,
        loop_penalty_coef=args.loop_penalty_coef,
        loop_window_base=args.loop_window_base,
        loop_window_per_len=args.loop_window_per_len,
        obs_mode=args.obs_mode,
    )
