import numpy as np
import glob

files = glob.glob('human_plays_*.npz')
if not files:
    print("No data files found.")
    exit()

f = files[0]
print(f"Analyzing {f}...")
data = np.load(f, allow_pickle=True)
states = data['states']
if 'games' in data:
    print(f"Metadata 'games' count: {data['games']}")
else:
    print("Metadata 'games' count not found.")

# Old Method: Length Drop
old_games = 0
STARTING_LEN = 0.03
for i in range(1, len(states)):
    curr_len = states[i]['scalar'][0]
    prev_len = states[i-1]['scalar'][0]
    
    # Original logic from train_imitation.py
    is_reset = (curr_len <= STARTING_LEN + 0.005 and prev_len > curr_len + 0.01)
    if is_reset:
        old_games += 1

print(f"Old Method (Length Drop) detected: {old_games + 1} games")

# New Method: Head at Center + Length 3
new_games = 0
# Center is (5, 5). Padded is (6, 6).
# CNN shape is (4, 12, 12). Channel 0 is Head.
# We check if this frame looks like a start frame.
# A start frame has Length ~0.03 and Head at (6, 6).
# Note: The very first frame is always a start frame.

start_indices = []
for i in range(len(states)):
    s = states[i]
    length = s['scalar'][0]
    # Check length is close to 0.03
    if abs(length - 0.03) < 0.001:
        # Check head at (6, 6)
        # cnn is (4, 12, 12)
        # channel 0 is head
        if s['cnn'][0, 6, 6] == 1.0:
            # This is a start frame.
            # But we only want to count it if it's a NEW start.
            # Since the snake stays at start for a bit, we might see multiple consecutive frames.
            # We count a new game if the PREVIOUS frame was NOT a start frame (or if i==0).
            
            is_start_sequence = True
            if i > 0:
                prev_s = states[i-1]
                prev_len = prev_s['scalar'][0]
                # If previous frame was also length 3 and head at center, it's the same game start.
                if abs(prev_len - 0.03) < 0.001 and prev_s['cnn'][0, 6, 6] == 1.0:
                    is_start_sequence = False
            
            if is_start_sequence:
                new_games += 1
                start_indices.append(i)

print(f"New Method (Head Reset) detected: {new_games} games")

# Let's see the scores of these games
scores = []
for k in range(len(start_indices)):
    start = start_indices[k]
    end = start_indices[k+1] if k+1 < len(start_indices) else len(states)
    
    game_states = states[start:end]
    if len(game_states) > 0:
        max_len = max(s['scalar'][0] for s in game_states)
        score = int(max_len * 100 + 0.001) - 3  # Add epsilon for float precision
        scores.append(score)

print(f"Scores count: {len(scores)}")
print(f"Count of Score < 0: {sum(1 for s in scores if s < 0)}")
print(f"Count of Score 0: {scores.count(0)}")
print(f"Count of Score 1: {scores.count(1)}")
