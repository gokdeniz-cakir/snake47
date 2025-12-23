import numpy as np
import glob
import sys

files = glob.glob('human_plays_*.npz')
if not files:
    print("No data files found.")
    sys.exit()

f = files[0]
print(f"Reading {f}...")
data = np.load(f, allow_pickle=True)
if 'games' in data:
    print(f"METADATA_GAMES_COUNT: {data['games']}")
else:
    print("METADATA_GAMES_COUNT: NOT FOUND")
