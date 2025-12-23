import numpy as np
import glob

# Find all data files
files = glob.glob('human_plays_*.npz')
print(f'Found {len(files)} files\n')

total_games = 0
total_frames = 0

for f in files:
    data = np.load(f, allow_pickle=True)
    games = data['games'].item() if 'games' in data else 'unknown'
    frames = len(data['states'])
    total_games += games if isinstance(games, int) else 0
    total_frames += frames
    print(f'{f}:')
    print(f'  Games: {games}')
    print(f'  Frames: {frames}')
    print()

print(f'TOTAL: {total_games} games, {total_frames} frames')
