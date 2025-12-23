
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def plot_progress(log_dir='evolution/logs', plot_dir='results'):
    # Find latest log file
    list_of_files = glob.glob(os.path.join(log_dir, '*.csv'))
    if not list_of_files:
        print("No log files found!")
        return
        
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"Plotting data from: {latest_file}")
    
    # Create plot directory
    os.makedirs(plot_dir, exist_ok=True)
    
    # Load data
    df = pd.read_csv(latest_file)
    
    # Use a nice style
    plt.style.use('ggplot')
    
    # 1. Score Progression
    plt.figure(figsize=(12, 6))
    
    # Plot average and median
    plt.plot(df['generation'], df['avg_score'], label='Avg Score', color='blue', alpha=0.6)
    if 'median_score' in df.columns:
        plt.plot(df['generation'], df['median_score'], label='Median Score', color='orange', alpha=0.6, linestyle='--')
        
    # Plot best (5-game avg)
    plt.plot(df['generation'], df['best_score'], label='Best Agent (5-game Avg)', color='green', linewidth=2)
    
    # Plot single game best (cumulative max)
    if 'single_best_score' in df.columns:
        df['cumulative_single_best_score'] = df['single_best_score'].cummax()
        plt.plot(df['generation'], df['cumulative_single_best_score'], label='Single Best Game (Cumulative)', color='red', linewidth=2, alpha=0.8)
    elif 'max_score' in df.columns: # Fallback for older logs
         df['cumulative_max_score'] = df['max_score'].cummax()
         plt.plot(df['generation'], df['cumulative_max_score'], label='Max Score (Cumulative)', color='red', linewidth=2, alpha=0.8)

    plt.title(f"Score Progression (ES Agent) - {os.path.basename(latest_file)}")
    plt.xlabel("Generation")
    plt.ylabel("Score (Apples Eaten)")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    save_path = os.path.join(plot_dir, 'score_history.png')
    plt.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")
    plt.close()
    
    # 2. Coverage Progression
    plt.figure(figsize=(12, 6))
    
    # Convert to percentage
    df['avg_coverage_pct'] = df['avg_coverage'] * 100
    df['best_coverage_pct'] = df['best_coverage'] * 100
    
    plt.plot(df['generation'], df['avg_coverage_pct'], label='Avg Coverage %', color='blue', alpha=0.6)
    plt.plot(df['generation'], df['best_coverage_pct'], label='Best Agent Coverage %', color='green', linewidth=2)
    
    if 'single_best_coverage' in df.columns:
        df['single_best_coverage_pct'] = df['single_best_coverage'] * 100
        df['cumulative_single_best_coverage_pct'] = df['single_best_coverage_pct'].cummax()
        plt.plot(df['generation'], df['cumulative_single_best_coverage_pct'], label='Single Best Game % (Cumulative)', color='red', linewidth=2, alpha=0.8)
    
    plt.title(f"Board Coverage % (10x10 Grid)")
    plt.xlabel("Generation")
    plt.ylabel("Coverage %")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    save_path = os.path.join(plot_dir, 'coverage_history.png')
    plt.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")
    plt.close()

if __name__ == "__main__":
    plot_progress()
