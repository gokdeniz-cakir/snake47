import matplotlib.pyplot as plt
import re

def parse_data(filename):
    steps = []
    train_avgs = []
    eval_scores = []
    epsilons = []

    with open(filename, 'r') as f:
        for line in f:
            # Regex to match: Step 5000 | Train Avg: 0.2 | Eval Score: 0.0 | Eps: 0.95
            match = re.search(r"Step (\d+) \| Train Avg: ([\d\.]+) \| Eval Score: ([\d\.]+) \| Eps: ([\d\.]+)", line)
            if match:
                steps.append(int(match.group(1)))
                train_avgs.append(float(match.group(2)))
                eval_scores.append(float(match.group(3)))
                epsilons.append(float(match.group(4)))
    
    return steps, train_avgs, eval_scores, epsilons

def plot_data(steps, train_avgs, eval_scores, epsilons):
    fig, ax1 = plt.figure(figsize=(12, 6)), plt.gca()

    # Plot Scores
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Score', color='black')
    l1, = ax1.plot(steps, train_avgs, label='Train Avg (100 games)', color='blue', alpha=0.6)
    l2, = ax1.plot(steps, eval_scores, label='Eval Score (5 games)', color='orange', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3)

    # Plot Epsilon on secondary axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Epsilon', color='green')
    l3, = ax2.plot(steps, epsilons, label='Epsilon', color='green', linestyle='--', alpha=0.4)
    ax2.tick_params(axis='y', labelcolor='green')

    # Title & Legend
    plt.title('Snake Agent Training Progress')
    lines = [l1, l2, l3]
    ax1.legend(lines, [l.get_label() for l in lines], loc='upper left')

    plt.tight_layout()
    plt.savefig('training_plot.png')
    print("Plot saved to training_plot.png")

if __name__ == "__main__":
    filename = "training_data.txt"
    try:
        steps, train, eval_s, eps = parse_data(filename)
        if not steps:
            print(f"No data found in {filename}. Make sure the format matches 'Step X | Train Avg: Y | ...'")
        else:
            print(f"Parsed {len(steps)} data points.")
            plot_data(steps, train, eval_s, eps)
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
