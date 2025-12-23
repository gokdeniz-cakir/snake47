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
if 'games' in data:
    print(f"Metadata 'games' count: {data['games']}")

# Combined Detection
# We iterate and check for ANY sign of reset.

games = 0
game_indices = [0]

# Trackers
prev_len = states[0]['scalar'][0]
s0 = states[0]['cnn'][0]
prev_h = np.unravel_index(np.argmax(s0), s0.shape)

for i in range(1, len(states)):
    s = states[i]
    curr_len = s['scalar'][0]
    
    # Head pos
    c = s['cnn'][0]
    h = np.unravel_index(np.argmax(c), c.shape)
    
    # 1. Length Drop (Original strict)
    # is_reset_len = (curr_len <= 0.035 and prev_len > curr_len + 0.01)
    # Let's relax it: Any drop to ~0.03
    is_reset_len = (curr_len <= 0.035 and prev_len > curr_len)
    
    # 2. Head at Center (Start state) - DISABLED
    # is_head_center = (h == (6, 6))
    # is_reset_head = (is_head_center and prev_h != (6, 6))
    is_reset_head = False
    
    # 3. Jump (Teleport)
    dist = abs(h[0] - prev_h[0]) + abs(h[1] - prev_h[1])
    is_reset_jump = (dist > 1)
    
    if is_reset_len or is_reset_jump:
        games += 1
        game_indices.append(i)
        # print(f"Reset at {i}: Len {is_reset_len}, Head {is_reset_head}, Jump {is_reset_jump}")
    
    prev_len = curr_len
    prev_h = h

print(f"Combined Method detected: {games + 1} games")
