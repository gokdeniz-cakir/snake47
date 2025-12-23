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

games = 0
game_indices = [0]
in_start_sequence = True # First frame is start

# Start Body Positions (Padded)
# Head: (6, 6)
# Body 1: (5, 6)
# Body 2: (4, 6)

for i in range(1, len(states)):
    s = states[i]
    
    # Check Head
    c = s['cnn']
    # Channel 0: Head
    if c[0, 6, 6] != 1.0:
        in_start_sequence = False
        continue
        
    # Check Body (Channel 2)
    # We expect non-zero at (5, 6) and (4, 6)
    # And zero elsewhere?
    # Actually, let's just check if (5, 6) and (4, 6) are occupied.
    # Values depend on length, but for length 3 start:
    # (5, 6) -> 1/3
    # (4, 6) -> 2/3
    
    b1 = c[2, 5, 6]
    b2 = c[2, 4, 6]
    
    is_start_body = (b1 > 0 and b2 > 0)
    
    if is_start_body:
        if not in_start_sequence:
            games += 1
            game_indices.append(i)
            in_start_sequence = True
    else:
        in_start_sequence = False

print(f"Body Reset Method detected: {games + 1} games")
