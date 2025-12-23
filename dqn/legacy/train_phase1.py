import os

import train_dqn
import train_phase2

PHASE1_STEPS = 250_000
SAVE_DIR = "checkpoints_phase1"
LOG_DIR = "logs_phase1"
RESUME_IF_EXISTS = True
AUTO_START_PHASE2 = True

PHASE1_ENV_OVERRIDES = {
    "start_length_min": 2,
    "start_length_max": 2,
    "step_cost": 0.01,
    "food_reward": 0.0,
    "tail_reward_coef": 0.0,
    "grow_on_food": False,
    "growth_interval": 25,
    "growth_max_length": 30,
    "stall_penalty_coef": 0.15,
    "loop_penalty_coef": 0.3,
    "render_growth_flash_frames": 4,
}

PHASE1_TRAIN_OVERRIDES = {
    "total_steps": PHASE1_STEPS,
    "save_dir": SAVE_DIR,
    "log_dir": LOG_DIR,
}

PHASE1_EVAL_OVERRIDES = {
    "start_length_min": 2,
    "start_length_max": 2,
}


def apply_overrides() -> None:
    train_dqn.ENV_CONFIG.update(PHASE1_ENV_OVERRIDES)
    train_dqn.TRAIN_CONFIG.update(PHASE1_TRAIN_OVERRIDES)
    train_dqn.EVAL_CONFIG.update(PHASE1_EVAL_OVERRIDES)


def run() -> None:
    apply_overrides()
    args = train_dqn.parse_args([])
    args.total_steps = PHASE1_STEPS
    args.save_dir = SAVE_DIR
    args.log_dir = LOG_DIR

    latest = os.path.join(SAVE_DIR, "checkpoint_latest.pth")
    if RESUME_IF_EXISTS and os.path.exists(latest):
        args.resume = latest

    train_dqn.train(args)

    if AUTO_START_PHASE2:
        init_from = os.path.join(SAVE_DIR, "model_best.pth")
        if not os.path.exists(init_from):
            init_from = None
        train_phase2.run(init_from=init_from)


if __name__ == "__main__":
    run()
