import numpy as np
import glob
import sys

files = glob.glob('human_plays_*.npz')
if not files:
    print("No data files found.")
    sys.exit()

f = files[0]
print(f"Analyzing {f}...")
data = np.load(f, allow_pickle=True)
states = data['states']

print(f"Total frames: {len(states)}")

# Extract head positions
# cnn shape: (N, 4, 12, 12)
# Channel 0 is head.
# We can use argmax to find the head position.

# To be efficient, we can iterate or use numpy ops.
# Since N ~ 10000, iteration is fine.

jumps = 0
jump_indices = []

# Get first head pos
s0 = states[0]['cnn'][0]
h0 = np.unravel_index(np.argmax(s0), s0.shape)
prev_h = h0

for i in range(1, len(states)):
    s = states[i]['cnn'][0]
    h = np.unravel_index(np.argmax(s), s.shape)
    
    # Calculate Manhattan distance
    dist = abs(h[0] - prev_h[0]) + abs(h[1] - prev_h[1])
    
    if dist > 1:
        jumps += 1
        jump_indices.append(i)
        # print(f"Jump at {i}: {prev_h} -> {h} (dist {dist})")
    
    prev_h = h

print(f"Detected {jumps} jumps (game resets).")
print(f"Total games = {jumps + 1}")

# Check scores of these games
scores = []
# First game: 0 to jump_indices[0]
# Last game: jump_indices[-1] to end
boundaries = [0] + jump_indices + [len(states)]

for k in range(len(boundaries) - 1):
    start = boundaries[k]
    end = boundaries[k+1]
    
    game_states = states[start:end]
    if len(game_states) > 0:
        max_len = max(s['scalar'][0] for s in game_states)
        score = int(max_len * 100 + 0.001) - 3
        scores.append(score)

print(f"Scores count: {len(scores)}")
print(f"Count of Score 0: {scores.count(0)}")
print(f"Count of Score 1: {scores.count(1)}")
print(f"Count of Score 2: {scores.count(2)}")
