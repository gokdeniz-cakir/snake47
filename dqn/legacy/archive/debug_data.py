import numpy as np
import glob

# Find all data files
files = glob.glob('human_plays_*.npz')
print(f'Found {len(files)} files')

total_frames = 0
for f in files:
    data = np.load(f, allow_pickle=True)
    states = data['states']
    total_frames += len(states)
    print(f'{f}: {len(states)} frames')

print(f'\nTotal frames across all files: {total_frames}')

# Look at the first file to understand game boundaries
if files:
    print(f'\n--- Analyzing {files[0]} ---')
    data = np.load(files[0], allow_pickle=True)
    states = data['states']
    
    print('\nFirst 30 normalized lengths (scalar[0]):')
    for i in range(min(30, len(states))):
        length = states[i]['scalar'][0]
        print(f'{i:3d}: {length:.4f}')
    
    # Count game resets
    print('\n--- Detecting game boundaries ---')
    game_count = 0
    for i in range(1, len(states)):
        curr_len = states[i]['scalar'][0]
        prev_len = states[i-1]['scalar'][0]
        
        # Current detection logic
        is_reset = curr_len < 0.05 and prev_len > curr_len + 0.02
        
        if is_reset:
            game_count += 1
            print(f'Game {game_count} ended at frame {i}: {prev_len:.4f} -> {curr_len:.4f}')
    
    print(f'\nTotal games detected in {files[0]}: {game_count + 1}')
