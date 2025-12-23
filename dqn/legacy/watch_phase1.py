import train_phase1

train_phase1.apply_overrides()

import watch_live


def main() -> None:
    args = watch_live.parse_args()
    if args.save_dir == "checkpoints":
        args.save_dir = train_phase1.SAVE_DIR
    watch_live.run(
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


if __name__ == "__main__":
    main()
