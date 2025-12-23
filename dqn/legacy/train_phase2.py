import os
from typing import Optional

import train_dqn

PHASE2_STEPS = 250_000
SAVE_DIR = "checkpoints_phase2"
LOG_DIR = "logs_phase2"
RESUME_IF_EXISTS = True
INIT_FROM = None

PHASE2_ENV_OVERRIDES = {
    "start_length_min": 2,
    "start_length_max": 2,
    "step_cost": -0.05,
    "food_reward": 25.0,
    "tail_reward_coef": 0.0,
    "grow_on_food": True,
    "growth_interval": 0,
    "growth_max_length": None,
}

PHASE2_TRAIN_OVERRIDES = {
    "total_steps": PHASE2_STEPS,
    "save_dir": SAVE_DIR,
    "log_dir": LOG_DIR,
}

PHASE2_EVAL_OVERRIDES = {
    "start_length_min": 2,
    "start_length_max": 2,
}


def apply_overrides() -> None:
    train_dqn.ENV_CONFIG.update(PHASE2_ENV_OVERRIDES)
    train_dqn.TRAIN_CONFIG.update(PHASE2_TRAIN_OVERRIDES)
    train_dqn.EVAL_CONFIG.update(PHASE2_EVAL_OVERRIDES)


def run(init_from: Optional[str] = None) -> None:
    apply_overrides()
    args = train_dqn.parse_args([])
    args.total_steps = PHASE2_STEPS
    args.save_dir = SAVE_DIR
    args.log_dir = LOG_DIR

    if init_from is None:
        init_from = INIT_FROM

    latest = os.path.join(SAVE_DIR, "checkpoint_latest.pth")
    if RESUME_IF_EXISTS and os.path.exists(latest):
        args.resume = latest
    elif init_from:
        args.init_from = init_from

    train_dqn.train(args)


if __name__ == "__main__":
    run()
