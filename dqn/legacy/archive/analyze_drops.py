import numpy as np

# Load one file and look at the length pattern
data = np.load('human_plays_20251204_175710.npz', allow_pickle=True)
states = data['states']
actual_games = data['games'].item()

print(f'File has {actual_games} games and {len(states)} frames')
print(f'\nAnalyzing length changes to understand reset pattern:\n')

# Look at all significant drops
drops = []
for i in range(1, len(states)):
    curr_len = states[i]['scalar'][0]
    prev_len = states[i-1]['scalar'][0]
    drop = prev_len - curr_len
    
    if drop > 0.01:  # Any drop
        drops.append((i, prev_len, curr_len, drop))

print(f'Found {len(drops)} drops > 0.01:')
print('First 50 drops:')
for i, (idx, prev, curr, drop) in enumerate(drops[:50]):
    print(f'{i+1:3d}. Frame {idx:4d}: {prev:.4f} -> {curr:.4f} (drop: {drop:.4f})')

print(f'\n... (showing 50 of {len(drops)} total drops)')
print(f'\nExpected ~{actual_games} game boundaries')
