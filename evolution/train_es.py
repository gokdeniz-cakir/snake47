"""
Training script for Evolution Strategy Snake Agent.

Uses simple (μ, λ) evolution strategy:
1. Evaluate population
2. Select top μ (elites)
3. Generate λ offspring via mutation
4. Repeat
"""

import argparse
import csv
import os
import time
import sys
from datetime import datetime
from typing import List, Tuple
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add parent directory to path to find snake_env
sys.path.append(str(Path(__file__).parent.parent))

from es_agent import ESAgent, evaluate_agent


def evaluate_single(args: Tuple[np.ndarray, int, int, int, int, int]) -> Tuple[int, float, float, float, float, float]:
    """Evaluate a single agent (for parallel execution)."""
    weights, idx, num_games, max_steps, seed, hidden1, hidden2 = args
    agent = ESAgent(hidden1, hidden2)
    agent.set_weights(weights)
    avg_score, avg_length, avg_coverage, max_score, max_coverage = evaluate_agent(
        agent, num_games=num_games, max_steps=max_steps, seed=seed
    )
    return idx, avg_score, avg_length, avg_coverage, max_score, max_coverage


def train(args: argparse.Namespace) -> None:
    """Main training loop."""
    
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Setup CSV logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(args.log_dir, f'train_es_{timestamp}.csv')
    log_file = open(log_path, 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        'generation', 'best_score', 'best_coverage', 'avg_score', 'avg_coverage',
        'median_score', 'min_score', 'max_score', 'best_ever_score', 'best_ever_coverage',
        'elite_avg_score', 'sigma', 'gen_time', 'total_time', 'single_best_score', 'single_best_coverage'
    ])
    print(f"Logging to: {log_path}")
    training_start = time.time()
    
    # Initialize population
    print(f"Initializing population of {args.population_size} agents...")
    
    population: List[ESAgent] = []
    start_gen = 0
    best_ever_score = 0.0
    best_ever_coverage = 0.0
    best_ever_agent = None
    
    # Resume from checkpoint if requested
    if args.resume:
        # Use --resume-from if provided, otherwise default to best_agent.npz
        if args.resume_from:
            checkpoint_path = args.resume_from
        else:
            checkpoint_path = os.path.join(args.save_dir, "best_agent.npz")
        
        if os.path.exists(checkpoint_path):
            print(f"Resuming from {checkpoint_path}...")
            base_agent = ESAgent.load(checkpoint_path)
            # Create population from best agent + mutations
            population.append(base_agent)
            for _ in range(args.population_size - 1):
                population.append(base_agent.mutate(sigma=args.sigma))
            print(f"Loaded best agent and created {args.population_size} mutants")
        else:
            print(f"No checkpoint found at {checkpoint_path}, starting fresh")
            args.resume = False
    
    if not args.resume:
        for _ in range(args.population_size):
            agent = ESAgent(args.hidden1, args.hidden2)
            population.append(agent)
    
    print(f"Network: {population[0].input_size} -> {args.hidden1} -> {args.hidden2} -> 3")
    print(f"Parameters per agent: {population[0].num_params()}")
    
    sigma = args.sigma  # Initialize sigma for logging
    
    for gen in range(args.generations):
        gen_start = time.time()
        
        # Evaluate all agents
        if args.workers > 1:
            # Parallel evaluation
            eval_args = [
                (agent.get_weights(), i, args.games_per_eval, args.max_steps, 
                 args.eval_seed, args.hidden1, args.hidden2)
                for i, agent in enumerate(population)
            ]
            
            results = [None] * len(population)
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(evaluate_single, arg) for arg in eval_args]
                for future in as_completed(futures):
                    idx, score, length, coverage, max_sc, max_cov = future.result()
                    results[idx] = (score, length, coverage, max_sc, max_cov)
            
            fitness = [(r[0], r[1], r[2], i, r[3], r[4]) for i, r in enumerate(results)]
        else:
            # Sequential evaluation
            fitness = []
            for i, agent in enumerate(population):
                score, length, coverage, max_sc, max_cov = evaluate_agent(
                    agent, num_games=args.games_per_eval, 
                    max_steps=args.max_steps, seed=args.eval_seed
                )
                fitness.append((score, length, coverage, i, max_sc, max_cov))
        
        # Sort by score (primary) and coverage (secondary)
        fitness.sort(key=lambda x: (x[0], x[2]), reverse=True)
        
        # Stats
        scores = [f[0] for f in fitness]
        coverages = [f[2] for f in fitness]
        best_score = scores[0]
        best_coverage = coverages[0]
        avg_score = np.mean(scores)
        avg_coverage = np.mean(coverages)
        
        # Track best ever
        if best_score > best_ever_score or (best_score == best_ever_score and best_coverage > best_ever_coverage):
            best_ever_score = best_score
            best_ever_coverage = best_coverage
            best_ever_agent = population[fitness[0][3]].copy()
            
            # Save best agent
            save_path = os.path.join(args.save_dir, "best_agent.npz")
            best_ever_agent.save(save_path)
        
        gen_time = time.time() - gen_start
        total_time = time.time() - training_start
        
        # Additional stats for logging
        median_score = np.median(scores)
        min_score = min(scores)
        max_score = max(scores)
        elite_avg_score = np.mean(scores[:args.num_elites])
        
        # Single-game best (from max values)
        single_best_scores = [f[4] for f in fitness]  # max_score per agent
        single_best_coverages = [f[5] for f in fitness]  # max_coverage per agent
        single_best_score = max(single_best_scores)
        single_best_coverage = max(single_best_coverages)
        
        # Log to CSV
        log_writer.writerow([
            gen + 1, best_score, best_coverage, avg_score, avg_coverage,
            median_score, min_score, max_score, best_ever_score, best_ever_coverage,
            elite_avg_score, sigma, f'{gen_time:.2f}', f'{total_time:.2f}',
            single_best_score, single_best_coverage
        ])
        log_file.flush()
        
        print(f"Gen {gen+1:4d} | "
              f"Best: {best_score:5.1f} ({best_coverage*100:4.1f}%) | "
              f"Avg: {avg_score:5.1f} ({avg_coverage*100:4.1f}%) | "
              f"Ever: {best_ever_score:5.1f} ({best_ever_coverage*100:4.1f}%) | "
              f"Time: {gen_time:.1f}s")
        
        # Periodic checkpoint
        if (gen + 1) % args.checkpoint_interval == 0:
            ckpt_path = os.path.join(args.save_dir, f"gen_{gen+1:04d}.npz")
            population[fitness[0][3]].save(ckpt_path)
            print(f"  -> Saved checkpoint: {ckpt_path}")
        
        # Check for convergence
        if best_ever_coverage >= 0.95:
            print(f"\n🎉 Solved! Agent can fill {best_ever_coverage*100:.1f}% of the board!")
            break
        
        # Early stop check
        if gen >= 100 and best_ever_score < 5:
            print("\n⚠️  Poor progress after 100 generations. Consider adjusting hyperparameters.")
        
        # Select elites
        elite_indices = [f[3] for f in fitness[:args.num_elites]]
        elites = [population[i].copy() for i in elite_indices]
        
        # Adaptive mutation rate
        sigma = args.sigma
        if args.adaptive_sigma:
            # Decrease sigma if making progress, increase if stuck
            if gen > 10:
                recent_improvement = best_score - scores[args.num_elites]
                if recent_improvement < 0.5:
                    sigma = min(args.sigma * 2, 0.5)
                else:
                    sigma = max(args.sigma * 0.8, 0.01)
        
        # Generate new population
        new_population = []
        
        # Keep elites
        new_population.extend(elites)
        
        # Generate offspring via mutation
        while len(new_population) < args.population_size:
            # Pick a random elite to mutate
            parent = elites[np.random.randint(len(elites))]
            child = parent.mutate(sigma=sigma)
            new_population.append(child)
        
        population = new_population
    
    # Final save
    if best_ever_agent is not None:
        final_path = os.path.join(args.save_dir, "final_best.npz")
        best_ever_agent.save(final_path)
        print(f"\n✓ Training complete. Best agent saved to: {final_path}")
        print(f"  Best score: {best_ever_score:.1f}, Coverage: {best_ever_coverage*100:.1f}%")
    
    log_file.close()
    print(f"  Training log saved to: {log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Snake agent via Evolution Strategy")
    
    # Population
    parser.add_argument("--population-size", type=int, default=100,
                        help="Number of agents in population")
    parser.add_argument("--num-elites", type=int, default=20,
                        help="Number of top agents to keep each generation")
    parser.add_argument("--generations", type=int, default=500,
                        help="Number of generations to evolve")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from checkpoint")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to checkpoint file to resume from (default: best_agent.npz)")
    
    # Network
    parser.add_argument("--hidden1", type=int, default=32,
                        help="First hidden layer size")
    parser.add_argument("--hidden2", type=int, default=32,
                        help="Second hidden layer size")
    
    # Evaluation
    parser.add_argument("--games-per-eval", type=int, default=5,
                        help="Games to average for fitness")
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="Max steps per game")
    parser.add_argument("--eval-seed", type=int, default=None,
                        help="Seed for reproducible evaluation")
    
    # Mutation
    parser.add_argument("--sigma", type=float, default=0.1,
                        help="Mutation standard deviation")
    parser.add_argument("--adaptive-sigma", action="store_true",
                        help="Adapt mutation rate based on progress")
    
    # Parallelization
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers for evaluation")
    
    # Saving
    parser.add_argument("--save-dir", type=str, default="evolution/checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--log-dir", type=str, default="evolution/logs",
                        help="Directory to save training logs")
    parser.add_argument("--checkpoint-interval", type=int, default=50,
                        help="Save checkpoint every N generations")
    
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
