"""
Snake Imitation Learning - Train an agent to play like you!

Uses behavioral cloning: supervised learning on (state, action) pairs.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# NETWORK ARCHITECTURES
# ==========================================

class ImitationCNN(nn.Module):
    """
    CNN that mimics your playstyle.
    Same architecture as DuelingDQN but outputs action probabilities.
    """
    def __init__(self):
        super().__init__()
        
        # CNN feature extractor
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        
        # Flatten: 64 * 8 * 8 = 4096
        self.flatten_size = 64 * 8 * 8
        
        # Fully connected (including 3 scalar inputs)
        self.fc1 = nn.Linear(self.flatten_size + 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 3)  # 3 actions: left, straight, right
    
    def forward(self, x, scalars):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        
        x = torch.cat((x, scalars), dim=1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Logits (softmax applied in loss)
        
        return x
    
    def predict(self, x, scalars):
        """Get action prediction."""
        with torch.no_grad():
            logits = self.forward(x, scalars)
            return logits.argmax(dim=1)
    
    def save(self, path='imitation_cnn.pth'):
        torch.save(self.state_dict(), path)
    
    def load(self, path='imitation_cnn.pth'):
        self.load_state_dict(torch.load(path))


class ImitationMLP(nn.Module):
    """
    Simple MLP version using 11 hand-crafted features.
    """
    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Linear(11, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 3)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def predict(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            return logits.argmax(dim=1)
    
    def save(self, path='imitation_mlp.pth'):
        torch.save(self.state_dict(), path)
    
    def load(self, path='imitation_mlp.pth'):
        self.load_state_dict(torch.load(path))


# ==========================================
# DATASET
# ==========================================

class SnakeDataset(Dataset):
    """Dataset for recorded human plays."""
    
    def __init__(self, data_files, model_type='cnn', min_score=0):
        """
        Args:
            data_files: list of .npz files from recording
            model_type: 'cnn' or 'mlp'
            min_score: minimum game score to include (filters out bad games)
                      Score = max snake length - 3 (starting length)
                      Default 0 includes all games.
        """
        self.model_type = model_type
        
        all_states = []
        all_scalars = []
        all_actions = []
        
        games_loaded = 0
        games_filtered = 0
        
        STARTING_LEN = 0.03  # Normalized starting length (3 blocks / 100 cells)
        
        for f in data_files:
            data = np.load(f, allow_pickle=True)
            states = data['states']
            actions = data['actions']
            
            # Split into individual games by detecting resets
            # A reset is when:
            # 1. Head is at center (6, 6) AND Length is small (~0.03)
            # 2. AND we weren't in this state in the previous frame (to count 1 per game)
            
            game_start = 0
            
            # Helper to check if state is a "start state"
            def is_start_state(state, model_type):
                # Check length
                length = state['scalar'][0]
                if length > 0.04: # Allow small float error, start is 0.03
                    return False
                
                # Check head pos if CNN
                if model_type == 'cnn':
                    s = state['cnn'][0]
                    head_pos = np.unravel_index(np.argmax(s), s.shape)
                    if head_pos != (6, 6):
                        return False
                
                return True

            prev_is_start = False
            if len(states) > 0:
                prev_is_start = is_start_state(states[0], model_type)
            
            for i in range(1, len(states)):
                curr_is_start = is_start_state(states[i], model_type)
                
                is_reset = False
                
                # Detect transition: Not Start -> Start
                if curr_is_start and not prev_is_start:
                    is_reset = True
                
                # Also detect "Jump" resets where we might miss the exact start frame
                # but the head jumps significantly.
                # (Optional, but good for robustness if start frame is dropped)
                
                prev_is_start = curr_is_start
                
                is_last = (i == len(states) - 1)
                
                if is_reset or is_last:
                    game_end = i if is_reset else i + 1
                    
                    game_states = states[game_start:game_end]
                    game_actions = actions[game_start:game_end]
                    
                    if len(game_states) > 0:
                        # Calculate score: max length reached - starting length
                        max_len = max(s['scalar'][0] for s in game_states)
                        score = int(max_len * 100) - 3
                        
                        if score >= min_score:
                            for state, action in zip(game_states, game_actions):
                                if model_type == 'cnn':
                                    all_states.append(state['cnn'])
                                    all_scalars.append(state['scalar'])
                                else:
                                    all_states.append(state['mlp'])
                                all_actions.append(action)
                            games_loaded += 1
                        else:
                            games_filtered += 1
                    
                    game_start = i
        
        self.states = np.array(all_states)
        self.actions = np.array(all_actions)
        
        if model_type == 'cnn':
            self.scalars = np.array(all_scalars)
        
        print(f"Loaded {len(self.states)} frames from {games_loaded} games")
        print(f"Filtered out {games_filtered} games with score < {min_score}")
        
        # Show action distribution
        if len(self.actions) > 0:
            unique, counts = np.unique(self.actions, return_counts=True)
            print("Action distribution:")
            action_names = ['Left', 'Straight', 'Right']
            for a, c in zip(unique, counts):
                print(f"  {action_names[a]}: {c} ({100*c/len(self.actions):.1f}%)")
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        if self.model_type == 'cnn':
            return (
                torch.tensor(self.states[idx], dtype=torch.float),
                torch.tensor(self.scalars[idx], dtype=torch.float),
                torch.tensor(self.actions[idx], dtype=torch.long)
            )
        else:
            return (
                torch.tensor(self.states[idx], dtype=torch.float),
                torch.tensor(self.actions[idx], dtype=torch.long)
            )


# ==========================================
# TRAINING
# ==========================================

def train_imitation(model_type='cnn', epochs=100, batch_size=32, lr=1e-3, min_score=0):
    """Train imitation model on recorded data."""
    
    # Find all data files
    data_files = glob.glob('human_plays_*.npz')
    
    if not data_files:
        print("No data files found! Run record_human.py first.")
        return None
    
    print(f"Found {len(data_files)} data file(s)")
    
    # Create dataset and loader
    dataset = SnakeDataset(data_files, model_type=model_type, min_score=min_score)
    
    # Split into train/val (90/10)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\nTrain: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    # Create model
    if model_type == 'cnn':
        model = ImitationCNN().to(device)
    else:
        model = ImitationMLP().to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Class weights to handle imbalanced actions (usually lots of 'straight')
    action_counts = np.bincount(dataset.actions, minlength=3)
    weights = 1.0 / (action_counts + 1)
    weights = weights / weights.sum() * 3  # Normalize
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    print(f"\nTraining {model_type.upper()} model...")
    print("="*50)
    
    best_val_acc = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            if model_type == 'cnn':
                states, scalars, actions = batch
                states, scalars, actions = states.to(device), scalars.to(device), actions.to(device)
                logits = model(states, scalars)
            else:
                states, actions = batch
                states, actions = states.to(device), actions.to(device)
                logits = model(states)
            
            loss = criterion(logits, actions)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += (logits.argmax(dim=1) == actions).sum().item()
            train_total += len(actions)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if model_type == 'cnn':
                    states, scalars, actions = batch
                    states, scalars, actions = states.to(device), scalars.to(device), actions.to(device)
                    logits = model(states, scalars)
                else:
                    states, actions = batch
                    states, actions = states.to(device), actions.to(device)
                    logits = model(states)
                
                val_correct += (logits.argmax(dim=1) == actions).sum().item()
                val_total += len(actions)
        
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {train_loss/len(train_loader):.4f} | "
                  f"Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save(f'imitation_{model_type}_best.pth')
    
    print("="*50)
    print(f"Best validation accuracy: {best_val_acc:.1f}%")
    print(f"Model saved to imitation_{model_type}_best.pth")
    
    return model


# ==========================================
# DATA AUGMENTATION (Optional)
# ==========================================

def augment_data(states, actions):
    """
    Augment data by mirroring.
    If you turn left, the mirror would turn right.
    
    This doubles your dataset!
    """
    augmented_states = []
    augmented_actions = []
    
    for state, action in zip(states, actions):
        # Original
        augmented_states.append(state)
        augmented_actions.append(action)
        
        # Mirrored (flip horizontally)
        if isinstance(state, dict):
            mirrored = {
                'cnn': np.flip(state['cnn'], axis=2).copy(),  # Flip width
                'scalar': state['scalar'].copy(),
                'mlp': state['mlp'].copy()  # Would need proper mirroring
            }
            # Mirror the action: left <-> right, straight stays
            if action == 0:
                mirrored_action = 2
            elif action == 2:
                mirrored_action = 0
            else:
                mirrored_action = action
            
            augmented_states.append(mirrored)
            augmented_actions.append(mirrored_action)
    
    return augmented_states, augmented_actions


# ==========================================
# MAIN
# ==========================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train imitation learning agent')
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'mlp'],
                        help='Model type: cnn or mlp')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--min_score', type=int, default=0,
                        help='Minimum game score to include (default 0 = all games)')
    
    args = parser.parse_args()
    
    train_imitation(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        min_score=args.min_score
    )