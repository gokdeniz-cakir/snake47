"""
Watch Agent 47 (trained ES agent) play Snake.
"""

import argparse
import json
import time
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path to find snake_env
sys.path.append(str(Path(__file__).parent.parent))

from es_agent import ESAgent, get_features
from snake_env import SnakeEnv


def watch(args: argparse.Namespace) -> None:
    """Watch the agent play."""
    
    # Load agent
    print(f"\n Loading snake47 from: {args.checkpoint}")
    agent = ESAgent.load(args.checkpoint)
    print(f"   Parameters: {agent.num_params()}")
    print(f"   Network: {agent.input_size} -> {agent.hidden1} -> {agent.hidden2} -> 3")
    
    # Prepare logging
    log_data = []
    
    total_score = 0
    total_coverage = 0
    
    for episode in range(args.episodes):
        env = SnakeEnv(
            width=10, 
            height=10,
            render_mode="human" if not args.headless else None,
            render_fps=args.fps,
            render_cell_size=args.cell_size,  # Bigger cells!
            seed=args.seed + episode if args.seed else None,
            start_length_min=2,
            start_length_max=2
        )
        
        print(f"\n{'='*40}")
        print(f"  Episode {episode + 1} / {args.episodes}")
        print(f"{'='*40}")
        
        steps = 0
        episode_data = {"steps": [], "score": 0, "length": 0, "death_reason": None}
        
        while not env.done and steps < args.max_steps:
            if not args.headless:
                env.render()
            
            action = agent.act(env, deterministic=not args.stochastic)
            _, _, reward, done, info = env.step(action)
            steps += 1
            
            # Log step data
            if args.log:
                episode_data["steps"].append({
                    "step": steps,
                    "action": action,
                    "score": env.score,
                    "length": len(env.snake),
                    "head": env.snake[0],
                    "food": env.food
                })
            
            if args.verbose:
                features = get_features(env)
                print(f"Step {steps}: action={action}, score={env.score}, "
                      f"length={len(env.snake)}, danger={features[:3]}")
        
        # Episode summary
        coverage = len(env.snake) / (env.width * env.height) * 100
        death_reason = info.get('death_reason', 'unknown')
        
        print(f"\n Results:")
        print(f"     Score:    {env.score}")
        print(f"     Length:   {len(env.snake)}")
        print(f"     Coverage: {coverage:.1f}%")
        print(f"     Steps:    {steps}")
        
        # Death reason with emoji
        death_emoji = {"wall": "🧱", "self": "🐍", "timeout": "⏰"}.get(death_reason, "❓")
        print(f"     Death:    {death_emoji} {death_reason.upper()}")
        
        total_score += env.score
        total_coverage += coverage
        
        # Save episode data
        if args.log:
            episode_data["score"] = env.score
            episode_data["length"] = len(env.snake)
            episode_data["coverage"] = coverage
            episode_data["death_reason"] = death_reason
            episode_data["total_steps"] = steps
            log_data.append(episode_data)
        
        env.close()
        
        if episode < args.episodes - 1:
            time.sleep(0.5)
    
    # Final summary
    print(f"\n{'='*40}")
    print(f"  snake47 Summary")
    print(f"{'='*40}")
    print(f"  Avg Score:    {total_score / args.episodes:.1f}")
    print(f"  Avg Coverage: {total_coverage / args.episodes:.1f}%")
    
    # Save log if requested
    if args.log:
        log_path = args.log_path or f"agent47_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        print(f"\n  📝 Log saved to: {log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch snake47 play Snake")
    
    parser.add_argument("--checkpoint", type=str, default="evolution/checkpoints/best_agent.npz",
                        help="Path to agent checkpoint")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to watch")
    parser.add_argument("--fps", type=int, default=15,
                        help="Frames per second for rendering")
    parser.add_argument("--cell-size", type=int, default=40,
                        help="Size of each cell in pixels (default 40 for bigger display)")
    parser.add_argument("--max-steps", type=int, default=2000,
                        help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic action selection")
    parser.add_argument("--headless", action="store_true",
                        help="Run without rendering (console only)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print step-by-step info")
    parser.add_argument("--log", action="store_true",
                        help="Log episode data to JSON for visualization")
    parser.add_argument("--log-path", type=str, default=None,
                        help="Custom path for log file")
    
    return parser.parse_args()


if __name__ == "__main__":
    watch(parse_args())
