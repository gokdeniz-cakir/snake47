import numpy as np

# Load one file
data = np.load('human_plays_20251204_175710.npz', allow_pickle=True)
states = data['states']
actions = data['actions']
actual_games = data['games'].item()
total_frames = data['frames'].item()

print(f'File metadata:')
print(f'  Games: {actual_games}')
print(f'  Frames (metadata): {total_frames}')
print(f'  Frames (actual): {len(states)}')
print(f'  Average frames per game: {len(states) / actual_games:.1f}')

# The issue: games are recorded continuously without explicit boundaries!
# We need to look at the actual game over conditions

# Let's check if there are any patterns in the data
# One approach: look for very short games (quick deaths)
print(f'\nLooking at length sequence (first 200 frames):')
for i in range(min(200, len(states))):
    length = states[i]['scalar'][0]
    if i == 0 or i == len(states) - 1:
        print(f'{i:4d}: {length:.4f}')
    elif i > 0:
        prev_len = states[i-1]['scalar'][0]
        if abs(length - prev_len) > 0.005:  # Any change
            print(f'{i:4d}: {length:.4f} (change: {length - prev_len:+.4f})')
