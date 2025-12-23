import argparse
import csv
import os
import random
import time
import sys
from collections import Counter, deque
from typing import Dict, Optional, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

# Add parent directory to path to find snake_env
sys.path.append(str(Path(__file__).parent.parent))

from dqn_model import DuelingDQN
from replay_buffer import PrioritizedReplayBuffer
from snake_env import SnakeEnv

ENV_CONFIG = {
    "width": 10,
    "height": 10,
    "obs_mode": "gradient",
    "start_length_min": 2,
    "start_length_max": 2,
    "step_cost": -0.05,
    "food_reward": 25.0,
    "death_penalty": -10.0,
    "timeout_factor": 25,
    "timeout_penalty": -11.0,
    "grow_on_food": True,
    "growth_interval": 0,
    "growth_max_length": None,
    "stall_penalty_coef": 0.10,
    "loop_penalty_coef": 0.2,
    "loop_window_base": 0,
    "loop_window_per_len": 2,
    "body_penalty_coef": 0.05,
    "body_penalty_power": 2.0,
    "tail_reward_coef": 0.01,
    "exclude_tail_from_penalty": True,
    "render_growth_flash_frames": 0,
}

TRAIN_CONFIG = {
    "total_steps": 500_000,
    "batch_size": 64,
    "lr": 1e-4,
    "gamma": 0.98,
    "n_step": 1,
    "replay_capacity": 100_000,
    "warmup_steps": 5_000,
    "train_every": 1,
    "target_update": 2_000,
    "per_alpha": 0.6,
    "per_beta_start": 0.4,
    "per_beta_frames": 500_000,
    "grad_clip": 1.0,
    "checkpoint_interval": 50_000,
    "log_interval": 2_000,
    "save_dir": "dqn/checkpoints",
    "log_dir": "dqn/logs",
    "seed": None,
}

EVAL_CONFIG = {
    "interval": 10_000,
    "episodes": 20,
    "seed": 123,
    "start_length_min": 2,
    "start_length_max": 2,
}


def make_env(
    seed: Optional[int],
    render_mode: Optional[str],
    timeout_factor: int,
    timeout_penalty: float,
    stall_penalty_coef: float,
    loop_penalty_coef: float,
    loop_window_base: int,
    loop_window_per_len: int,
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
        render_mode=render_mode,
        seed=seed,
    )


def beta_by_step(step: int, start: float, frames: int) -> float:
    if frames <= 0:
        return 1.0
    return min(1.0, start + (1.0 - start) * (step / frames))


def save_checkpoint(
    path: str,
    policy_net: DuelingDQN,
    target_net: DuelingDQN,
    optimizer: optim.Optimizer,
    step: int,
    episodes_done: int,
    best_eval: float,
    best_score: int,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "policy_state": policy_net.state_dict(),
            "target_state": target_net.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": step,
            "episodes": episodes_done,
            "best_eval": best_eval,
            "best_score": best_score,
            "args": vars(args),
        },
        path,
    )


def load_checkpoint(
    path: str,
    policy_net: DuelingDQN,
    target_net: DuelingDQN,
    optimizer: Optional[optim.Optimizer],
    device: torch.device,
) -> Tuple[int, int, float, int]:
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "policy_state" in checkpoint:
        policy_net.load_state_dict(checkpoint["policy_state"])
        target_state = checkpoint.get("target_state")
        if target_state:
            target_net.load_state_dict(target_state)
        else:
            target_net.load_state_dict(policy_net.state_dict())
        if optimizer is not None and checkpoint.get("optimizer_state"):
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        step = int(checkpoint.get("step", 0))
        episodes = int(checkpoint.get("episodes", 0))
        best_eval = float(checkpoint.get("best_eval", -float("inf")))
        best_score = int(checkpoint.get("best_score", 0))
        return step, episodes, best_eval, best_score

    policy_net.load_state_dict(checkpoint)
    target_net.load_state_dict(policy_net.state_dict())
    return 0, 0, -float("inf"), 0


def load_policy_only(
    path: str, policy_net: DuelingDQN, target_net: DuelingDQN, device: torch.device
) -> None:
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "policy_state" in checkpoint:
        state = checkpoint["policy_state"]
    else:
        state = checkpoint
    policy_net.load_state_dict(state)
    target_net.load_state_dict(state)


def optimize(
    policy_net: DuelingDQN,
    target_net: DuelingDQN,
    optimizer: optim.Optimizer,
    replay: PrioritizedReplayBuffer,
    batch_size: int,
    gamma_n: float,
    beta: float,
    device: torch.device,
    grad_clip: float,
) -> Optional[float]:
    if hasattr(policy_net, "reset_noise"):
        policy_net.reset_noise()
    if hasattr(target_net, "reset_noise"):
        target_net.reset_noise()

    batch = replay.sample(batch_size, beta)
    if batch is None:
        return None

    samples, indices, weights = batch
    states, scalars, actions, rewards, next_states, next_scalars, dones = zip(*samples)

    states_t = torch.tensor(np.stack(states), dtype=torch.float32, device=device)
    scalars_t = torch.tensor(np.stack(scalars), dtype=torch.float32, device=device)
    actions_t = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
    next_states_t = torch.tensor(np.stack(next_states), dtype=torch.float32, device=device)
    next_scalars_t = torch.tensor(np.stack(next_scalars), dtype=torch.float32, device=device)
    dones_t = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)
    weights_t = torch.tensor(weights, dtype=torch.float32, device=device).unsqueeze(1)

    current_q = policy_net(states_t, scalars_t).gather(1, actions_t)
    with torch.no_grad():
        next_actions = policy_net(next_states_t, next_scalars_t).argmax(1, keepdim=True)
        next_q = target_net(next_states_t, next_scalars_t).gather(1, next_actions)
        target_q = rewards_t + gamma_n * next_q * (1.0 - dones_t)

    td_errors = target_q - current_q
    loss = (weights_t * F.smooth_l1_loss(current_q, target_q, reduction="none")).mean()

    optimizer.zero_grad()
    loss.backward()
    if grad_clip:
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), grad_clip)
    optimizer.step()

    replay.update_priorities(indices, td_errors.detach().cpu().numpy().squeeze())
    return float(loss.item())


def evaluate(
    policy_net: DuelingDQN,
    device: torch.device,
    env_seed: Optional[int],
    timeout_factor: int,
    timeout_penalty: float,
    stall_penalty_coef: float,
    loop_penalty_coef: float,
    loop_window_base: int,
    loop_window_per_len: int,
    obs_mode: str,
    start_length_min: int,
    start_length_max: int,
    episodes: int = 5,
) -> Tuple[float, float, float, int, float, int, Dict[str, int]]:
    eval_env = make_env(
        seed=env_seed,
        render_mode=None,
        timeout_factor=timeout_factor,
        timeout_penalty=timeout_penalty,
        stall_penalty_coef=stall_penalty_coef,
        loop_penalty_coef=loop_penalty_coef,
        loop_window_base=loop_window_base,
        loop_window_per_len=loop_window_per_len,
        obs_mode=obs_mode,
        start_length_min=start_length_min,
        start_length_max=start_length_max,
    )
    scores = []
    returns = []
    lengths = []
    deaths: Counter = Counter()

    policy_net.eval()
    with torch.no_grad():
        for ep in range(episodes):
            if env_seed is not None:
                eval_env.seed(env_seed + ep)
            state, scalars = eval_env.reset()
            done = False
            total_reward = 0.0
            while not done:
                s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                sc = torch.tensor(scalars, dtype=torch.float32, device=device).unsqueeze(0)
                action = int(policy_net(s, sc).argmax(dim=1).item())
                state, scalars, reward, done, info = eval_env.step(action)
                total_reward += float(reward)
            scores.append(int(info["score"]))
            returns.append(total_reward)
            deaths[str(info.get("death_reason"))] += 1
            lengths.append(int(info.get("length", len(eval_env.snake))))

    policy_net.train()
    scores_arr = np.asarray(scores, dtype=np.float32)
    returns_arr = np.asarray(returns, dtype=np.float32)
    avg_score = float(scores_arr.mean()) if scores_arr.size else 0.0
    avg_reward = float(returns_arr.mean()) if returns_arr.size else 0.0
    median_score = float(np.median(scores_arr)) if scores_arr.size else 0.0
    max_score = int(scores_arr.max()) if scores_arr.size else 0
    lengths_arr = np.asarray(lengths, dtype=np.float32)
    avg_length = float(lengths_arr.mean()) if lengths_arr.size else 0.0
    max_length = int(lengths_arr.max()) if lengths_arr.size else 0
    return avg_score, avg_reward, median_score, max_score, avg_length, max_length, dict(deaths)


def train(args: argparse.Namespace) -> None:
    if args.n_step < 1:
        raise ValueError("--n-step must be >= 1")

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_seed = args.seed
    env = make_env(
        seed=env_seed,
        render_mode="human" if args.render else None,
        timeout_factor=args.timeout_factor,
        timeout_penalty=args.timeout_penalty,
        stall_penalty_coef=args.stall_penalty_coef,
        loop_penalty_coef=args.loop_penalty_coef,
        loop_window_base=args.loop_window_base,
        loop_window_per_len=args.loop_window_per_len,
        obs_mode=args.obs_mode,
        start_length_min=args.start_length_min,
        start_length_max=args.start_length_max,
    )

    policy_net = DuelingDQN(env.observation_shape, num_actions=3, scalar_dim=3).to(device)
    target_net = DuelingDQN(env.observation_shape, num_actions=3, scalar_dim=3).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.AdamW(policy_net.parameters(), lr=args.lr)
    replay = PrioritizedReplayBuffer(args.replay_capacity, alpha=args.per_alpha)

    resume_step = 0
    resume_episodes = 0
    best_eval = -float("inf")
    best_score = 0

    if args.resume:
        resume_step, resume_episodes, best_eval, best_score = load_checkpoint(
            args.resume, policy_net, target_net, optimizer, device
        )
    elif args.init_from:
        load_policy_only(args.init_from, policy_net, target_net, device)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.log_dir, f"train_{run_id}.csv")
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    log_fields = [
        "event",
        "step",
        "episodes",
        "episode_steps",
        "episode_score",
        "death_reason",
        "episode_length",
        "avg_score",
        "avg_loss",
        "best_score",
        "replay_size",
        "eval_score",
        "eval_score_median",
        "eval_score_max",
        "eval_reward",
        "eval_avg_length",
        "eval_max_length",
        "eval_death_self",
        "eval_death_wall",
        "eval_death_timeout",
        "best_eval",
    ]
    log_writer = csv.DictWriter(log_file, fieldnames=log_fields)
    log_writer.writeheader()

    recent_scores = deque(maxlen=100)
    recent_losses = deque(maxlen=100)
    episodes_done = resume_episodes
    episode_steps = 0
    n_step_buffer: deque = deque()
    gamma_n = args.gamma ** args.n_step

    state, scalars = env.reset()
    start_step = resume_step + 1
    end_step = resume_step + args.total_steps
    try:
        def build_n_step_transition(buffer: deque) -> tuple:
            ret = 0.0
            next_s = None
            next_sc = None
            n_done = False
            for i, (_, _, _, r, ns, nsc, d) in enumerate(buffer):
                ret += (args.gamma ** i) * float(r)
                next_s = ns
                next_sc = nsc
                if d:
                    n_done = True
                    break
            s0, sc0, a0 = buffer[0][0], buffer[0][1], buffer[0][2]
            return (s0, sc0, a0, ret, next_s, next_sc, n_done)

        for step in range(start_step, end_step + 1):
            if hasattr(policy_net, "reset_noise"):
                policy_net.reset_noise()

            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                sc = torch.tensor(scalars, dtype=torch.float32, device=device).unsqueeze(0)
                action = int(policy_net(s, sc).argmax(dim=1).item())

            next_state, next_scalars, reward, done, info = env.step(action)
            n_step_buffer.append(
                (state, scalars, action, float(reward), next_state, next_scalars, done)
            )

            if len(n_step_buffer) >= args.n_step:
                replay.add(build_n_step_transition(n_step_buffer))
                if not done:
                    n_step_buffer.popleft()
            elif done:
                replay.add(build_n_step_transition(n_step_buffer))

            if done:
                while len(n_step_buffer) > 1:
                    n_step_buffer.popleft()
                    replay.add(build_n_step_transition(n_step_buffer))
                n_step_buffer.clear()

            state, scalars = next_state, next_scalars
            episode_steps += 1
            if args.render:
                env.render()

            if len(replay) >= args.warmup_steps and step % args.train_every == 0:
                beta = beta_by_step(step, args.per_beta_start, args.per_beta_frames)
                loss = optimize(
                    policy_net,
                    target_net,
                    optimizer,
                    replay,
                    args.batch_size,
                    gamma_n,
                    beta,
                    device,
                    args.grad_clip,
                )
                if loss is not None:
                    recent_losses.append(loss)

            if step % args.target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                episodes_done += 1
                score = info["score"]
                death_reason = info.get("death_reason")
                episode_length = int(info.get("length", len(env.snake)))
                best_score = max(best_score, score)
                recent_scores.append(score)
                log_writer.writerow(
                    {
                        "event": "episode",
                        "step": step,
                        "episodes": episodes_done,
                        "episode_steps": episode_steps,
                        "episode_score": score,
                        "death_reason": death_reason,
                        "episode_length": episode_length,
                        "best_score": best_score,
                        "replay_size": len(replay),
                    }
                )
                log_file.flush()
                state, scalars = env.reset()
                episode_steps = 0

            if step % args.log_interval == 0:
                avg_score = sum(recent_scores) / len(recent_scores) if recent_scores else 0.0
                avg_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0.0
                print(
                    f"Step {step} | Avg Score: {avg_score:.2f} | Loss: {avg_loss:.4f}"
                )
                log_writer.writerow(
                    {
                        "event": "train",
                        "step": step,
                        "episodes": episodes_done,
                        "avg_score": avg_score,
                        "avg_loss": avg_loss,
                        "best_score": best_score,
                        "replay_size": len(replay),
                    }
                )
                log_file.flush()

            if step % args.eval_interval == 0:
                (
                    eval_score,
                    eval_reward,
                    eval_median,
                    eval_max,
                    eval_len_avg,
                    eval_len_max,
                    eval_deaths,
                ) = evaluate(
                    policy_net,
                    device,
                    env_seed=args.eval_seed,
                    timeout_factor=args.timeout_factor,
                    timeout_penalty=args.timeout_penalty,
                    stall_penalty_coef=args.stall_penalty_coef,
                    loop_penalty_coef=args.loop_penalty_coef,
                    loop_window_base=args.loop_window_base,
                    loop_window_per_len=args.loop_window_per_len,
                    obs_mode=args.obs_mode,
                    start_length_min=args.eval_start_length_min,
                    start_length_max=args.eval_start_length_max,
                    episodes=args.eval_episodes,
                )
                eval_self = int(eval_deaths.get("self", 0))
                eval_wall = int(eval_deaths.get("wall", 0))
                eval_timeout = int(eval_deaths.get("timeout", 0))
                print(
                    f"Eval @ {step} | "
                    f"Score: {eval_score:.2f} (med {eval_median:.2f}, max {eval_max}) | "
                    f"Reward: {eval_reward:.2f} | "
                    f"Len: {eval_len_avg:.1f} (max {eval_len_max}) | "
                    f"Deaths self/wall/timeout: {eval_self}/{eval_wall}/{eval_timeout}"
                )
                if eval_score > best_eval:
                    best_eval = eval_score
                    best_path = os.path.join(args.save_dir, "model_best.pth")
                    torch.save(policy_net.state_dict(), best_path)
                    print(f"New best model saved: {best_path}")
                log_writer.writerow(
                    {
                        "event": "eval",
                        "step": step,
                        "episodes": episodes_done,
                        "eval_score": eval_score,
                        "eval_score_median": eval_median,
                        "eval_score_max": eval_max,
                        "eval_reward": eval_reward,
                        "eval_avg_length": eval_len_avg,
                        "eval_max_length": eval_len_max,
                        "eval_death_self": eval_self,
                        "eval_death_wall": eval_wall,
                        "eval_death_timeout": eval_timeout,
                        "best_eval": best_eval,
                        "best_score": best_score,
                    }
                )
                log_file.flush()

            if step % args.checkpoint_interval == 0:
                ckpt_path = os.path.join(args.save_dir, f"checkpoint_step_{step}.pth")
                save_checkpoint(
                    ckpt_path,
                    policy_net,
                    target_net,
                    optimizer,
                    step,
                    episodes_done,
                    best_eval,
                    best_score,
                    args,
                )
                latest_path = os.path.join(args.save_dir, "checkpoint_latest.pth")
                save_checkpoint(
                    latest_path,
                    policy_net,
                    target_net,
                    optimizer,
                    step,
                    episodes_done,
                    best_eval,
                    best_score,
                    args,
                )
    finally:
        log_file.close()


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN Snake agent")
    parser.add_argument("--total-steps", type=int, default=TRAIN_CONFIG["total_steps"])
    parser.add_argument("--batch-size", type=int, default=TRAIN_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=TRAIN_CONFIG["lr"])
    parser.add_argument("--gamma", type=float, default=TRAIN_CONFIG["gamma"])
    parser.add_argument("--n-step", type=int, default=TRAIN_CONFIG["n_step"])
    parser.add_argument(
        "--obs-mode",
        type=str,
        default=ENV_CONFIG["obs_mode"],
        choices=["gradient", "tail"],
    )
    parser.add_argument("--start-length-min", type=int, default=ENV_CONFIG["start_length_min"])
    parser.add_argument("--start-length-max", type=int, default=ENV_CONFIG["start_length_max"])
    parser.add_argument("--replay-capacity", type=int, default=TRAIN_CONFIG["replay_capacity"])
    parser.add_argument("--warmup-steps", type=int, default=TRAIN_CONFIG["warmup_steps"])
    parser.add_argument("--train-every", type=int, default=TRAIN_CONFIG["train_every"])
    parser.add_argument("--target-update", type=int, default=TRAIN_CONFIG["target_update"])
    parser.add_argument("--eval-interval", type=int, default=EVAL_CONFIG["interval"])
    parser.add_argument("--eval-episodes", type=int, default=EVAL_CONFIG["episodes"])
    parser.add_argument("--eval-seed", type=int, default=EVAL_CONFIG["seed"])
    parser.add_argument(
        "--eval-start-length-min", type=int, default=EVAL_CONFIG["start_length_min"]
    )
    parser.add_argument(
        "--eval-start-length-max", type=int, default=EVAL_CONFIG["start_length_max"]
    )
    parser.add_argument(
        "--checkpoint-interval", type=int, default=TRAIN_CONFIG["checkpoint_interval"]
    )
    parser.add_argument("--log-interval", type=int, default=TRAIN_CONFIG["log_interval"])
    parser.add_argument("--timeout-factor", type=int, default=ENV_CONFIG["timeout_factor"])
    parser.add_argument(
        "--timeout-penalty", type=float, default=ENV_CONFIG["timeout_penalty"]
    )
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
    parser.add_argument("--per-alpha", type=float, default=TRAIN_CONFIG["per_alpha"])
    parser.add_argument("--per-beta-start", type=float, default=TRAIN_CONFIG["per_beta_start"])
    parser.add_argument("--per-beta-frames", type=int, default=TRAIN_CONFIG["per_beta_frames"])
    parser.add_argument("--grad-clip", type=float, default=TRAIN_CONFIG["grad_clip"])
    parser.add_argument("--save-dir", type=str, default=TRAIN_CONFIG["save_dir"])
    parser.add_argument("--log-dir", type=str, default=TRAIN_CONFIG["log_dir"])
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--init-from", type=str, default=None)
    parser.add_argument("--seed", type=int, default=TRAIN_CONFIG["seed"])
    parser.add_argument("--render", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
