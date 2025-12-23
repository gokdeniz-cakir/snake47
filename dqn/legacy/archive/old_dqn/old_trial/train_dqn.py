import os
import random
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from snake_env import SnakeEnv, ACTION_SET


@dataclass
class Transition:
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)


class CNNDQN(nn.Module):
    def __init__(self, input_shape: Tuple[int, int, int], num_actions: int):
        super(CNNDQN, self).__init__()
        # input_shape is (3, 10, 10)
        
        self.features = nn.Sequential(
            # Conv 1: Detects edges/corners
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Conv 2: Detects shapes/patterns
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        # Calculate size: 10x10 input remains 10x10 due to padding=1
        # Flatten size = 64 channels * 10 * 10 = 6400
        linear_input_size = 64 * 10 * 10
        
        self.fc = nn.Sequential(
            nn.Linear(linear_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1) # Flatten
        return self.fc(x)


def select_action(state: torch.Tensor, policy_net: CNNDQN, epsilon: float) -> int:
    if random.random() < epsilon:
        return random.choice(list(ACTION_SET))
    with torch.no_grad():
        # state is (1, C, H, W)
        q_values = policy_net(state)
        return int(torch.argmax(q_values, dim=1).item())


def optimize_model(
    buffer: ReplayBuffer,
    policy_net: CNNDQN,
    target_net: CNNDQN,
    optimizer: optim.Optimizer,
    batch_size: int,
    gamma: float,
    device: torch.device,
    grad_clip: float = 1.0,
):
    if len(buffer) < batch_size:
        return 0.0

    transitions = buffer.sample(batch_size)

    # Stack tensors: state is (Batch, C, H, W)
    state_batch = torch.cat([t.state for t in transitions]).to(device)
    action_batch = torch.tensor([t.action for t in transitions], device=device, dtype=torch.long)
    reward_batch = torch.tensor([t.reward for t in transitions], device=device, dtype=torch.float32)
    next_state_batch = torch.cat([t.next_state for t in transitions]).to(device)
    done_batch = torch.tensor([t.done for t in transitions], device=device, dtype=torch.float32)

    # Q(s, a)
    q_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)

    # Double DQN:
    # 1. Select best action a' using policy_net: argmax Q_policy(s', a')
    # 2. Evaluate Q_target(s', a')
    with torch.no_grad():
        next_actions = policy_net(next_state_batch).argmax(dim=1, keepdim=True)
        max_next_q = target_net(next_state_batch).gather(1, next_actions).squeeze(1)
        target = reward_batch + gamma * max_next_q * (1.0 - done_batch)

    loss = nn.functional.smooth_l1_loss(q_values, target)

    optimizer.zero_grad()
    loss.backward()
    if grad_clip:
        nn.utils.clip_grad_norm_(policy_net.parameters(), grad_clip)
    optimizer.step()

    return float(loss.item())


def train_dqn(
    num_episodes: int = 2000,
    batch_size: int = 128,
    gamma: float = 0.99,
    lr: float = 1e-4,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.9997, # Slower decay for CNN
    target_update: int = 500,
    replay_capacity: int = 100000,
    seed: int = 42,
    load_path: Optional[str] = None,
    save_path: Optional[str] = None,
    warmup: int = 1000,
    grad_clip: float = 1.0,
    epsilon: Optional[float] = None,
    episode_offset: int = 0,
):
    random.seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    width, height = 10, 10
    env = SnakeEnv(
        width=width,
        height=height,
        step_cost=-0.1, # Increased cost to punish looping
        allow_reverse=False,
        seed=seed,
        dist_reward_coef=0.1, # Shaping reward enabled
        food_reward=10.0,
        death_penalty=-10.0,
        max_steps=width * height * 50,
    )
    
    # Input shape: (3, H, W) for CNN
    input_shape = (3, height, width)
    num_actions = len(ACTION_SET)

    policy_net = CNNDQN(input_shape, num_actions).to(device)
    if load_path and os.path.exists(load_path):
        try:
            policy_net.load_state_dict(torch.load(load_path, map_location=device))
            print(f"Loaded model from {load_path}")
        except RuntimeError as e:
            print(f"⚠️ Architecture mismatch detected: {e}")
            print("Backing up old checkpoint and starting fresh...")
            backup_path = load_path.replace(".pt", "_legacy_mlp.pt")
            os.rename(load_path, backup_path)
            print(f"Renamed {load_path} to {backup_path}")
            # Force fresh start settings
            epsilon_start = 1.0
            epsilon = 1.0
        
    target_net = CNNDQN(input_shape, num_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    buffer = ReplayBuffer(replay_capacity)

    epsilon = epsilon if epsilon is not None else epsilon_start
    all_returns: List[float] = []
    best_return = -float('inf')  # Track best performance
    global_step = 0

    for episode in range(num_episodes):
        obs = env.reset(seed=seed + episode)
        # Obs is (3, H, W) numpy array
        state = torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
        
        done = False
        episode_reward = 0.0
        episode_loss: List[float] = []

        while not done:
            action = select_action(state, policy_net, epsilon)
            next_obs, reward, done, _ = env.step(action)
            episode_reward += reward

            next_state_tensor = torch.tensor(
                next_obs, device=device, dtype=torch.float32
            ).unsqueeze(0)
            
            buffer.push(state, action, reward, next_state_tensor, done)

            if len(buffer) >= max(warmup, batch_size):
                loss_val = optimize_model(
                    buffer, policy_net, target_net, optimizer, batch_size, gamma, device, grad_clip
                )
                episode_loss.append(loss_val)

            state = next_state_tensor
            global_step += 1

            if global_step % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

        all_returns.append(episode_reward)
        # Iterative Decay & Cyclic Exploration
        if epsilon > epsilon_end:
            epsilon *= epsilon_decay
            epsilon = max(epsilon_end, epsilon)
        #else:
            # Cycle back up to 0.1 to escape local optima
            #print(f"♻️ Epsilon reached {epsilon_end}. Cycling back to 0.06 for warm restart!")
            #epsilon = 0.06

        if (episode + 1) % 20 == 0:
            avg_return = sum(all_returns[-20:]) / min(len(all_returns), 20)
            avg_loss = sum(episode_loss) / len(episode_loss) if episode_loss else 0.0
            print(
                f"Episode {episode + 1:4d} | avg return (last 20): {avg_return:.3f} "
                f"| epsilon: {epsilon:.3f} | avg loss: {avg_loss:.4f}"
            )
            
            # Save best model
            if avg_return > best_return:
                best_return = avg_return
                if save_path:
                    best_path = save_path.replace(".pt", "_best.pt")
                    torch.save(policy_net.state_dict(), best_path)
                    print(f"New best model saved! Return: {best_return:.3f}")

        # Periodic checkpoint every 1000 episodes
        if (episode + 1) % 1000 == 0 and save_path:
            torch.save(policy_net.state_dict(), save_path)
            print(f"Checkpoint saved to {save_path}")

    if save_path:
        torch.save(policy_net.state_dict(), save_path)
        print(f"Saved model to {save_path}")

    return policy_net, all_returns


def evaluate(policy_net: CNNDQN, episodes: int = 5, seed: int = 999, width: int = 10, height: int = 10):
    env = SnakeEnv(
        width=width,
        height=height,
        step_cost=-0.01,
        allow_reverse=False,
        seed=seed,
        dist_reward_coef=0.1,
        food_reward=10.0,
        death_penalty=-10.0,
        max_steps=width * height * 50,
    )
    device = next(policy_net.parameters()).device
    returns: List[Tuple[int, float]] = []

    for ep in range(episodes):
        obs = env.reset(seed=seed + ep)
        state = torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
        done = False
        total_reward = 0.0

        while not done:
            # env.render()
            with torch.no_grad():
                action = int(torch.argmax(policy_net(state), dim=1).item())
            next_obs, reward, done, _ = env.step(action)
            state = torch.tensor(
                next_obs, device=device, dtype=torch.float32
            ).unsqueeze(0)
            total_reward += reward
            # time.sleep(0.1) # Optional: slow down render

        returns.append((env.score, total_reward))
        print(f"Eval episode {ep + 1}: score={env.score} total_reward={total_reward:.3f}")

    return returns


if __name__ == "__main__":
    CHECKPOINT = "dqn_snake_cnn.pt"
    NUM_EPISODES = 20000

    print(f"Training CNNDQN (Tiny CNN) on 10x10 Snake for {NUM_EPISODES} episodes. Press Ctrl+C to stop early.")
    policy_net = None
    try:
        # Check if checkpoint exists to resume
        if os.path.exists(CHECKPOINT):
            print(f"Resuming training from {CHECKPOINT}...")
            policy_net, returns = train_dqn(
                num_episodes=NUM_EPISODES, 
                save_path=CHECKPOINT,
                load_path=CHECKPOINT,
                epsilon_start=0.5 # Resume with high exploration (Epsilon Injection)
            )
        else:
            print("Starting fresh training...")
            policy_net, returns = train_dqn(
                num_episodes=NUM_EPISODES, 
                save_path=CHECKPOINT,
                load_path=None,
                epsilon_start=1.0
            )
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        if policy_net is not None:
            torch.save(policy_net.state_dict(), CHECKPOINT)
            print(f"Latest policy saved at {CHECKPOINT}")
    else:
        if returns:
            last_100 = returns[-100:]
            avg_last_100 = sum(last_100) / len(last_100)
            print(f"Training complete. Avg return over last {len(last_100)} episodes: {avg_last_100:.3f}")
        print("Evaluating greedy policy:")
        evaluate(policy_net, episodes=5)
