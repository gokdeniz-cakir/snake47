import torch
import numpy as np
import random
from snake_env import SnakeEnv, ACTION_SET
from train_dqn import ConvDQN

def verify():
    width, height = 10, 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on {device}")

    # Load Model
    input_shape = (3, height, width)
    num_actions = len(ACTION_SET)
    policy_net = ConvDQN(input_shape, num_actions).to(device)
    
    try:
        policy_net.load_state_dict(torch.load("dqn_snake_cnn.pt", map_location=device))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    policy_net.eval()

    # Run Episodes
    num_episodes = 20
    scores = []
    
    for ep in range(num_episodes):
        env = SnakeEnv(
            width=width, 
            height=height, 
            step_cost=-0.01, 
            death_penalty=-10.0, 
            food_reward=10.0,
            max_steps=400,
            seed=None # Random seed
        )
        
        obs = env.reset()
        done = False
        score = 0
        steps = 0
        
        while not done:
            state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            
            # Epsilon-greedy action
            if random.random() < 0.05:
                action = random.choice(list(ACTION_SET))
            else:
                with torch.no_grad():
                    q_values = policy_net(state)
                    action = int(torch.argmax(q_values, dim=1).item())
            
            obs, reward, done, _ = env.step(action)
            score = env.score
            steps += 1
            
        scores.append(score)
        print(f"Episode {ep+1}: Score {score}, Steps {steps}")

    print(f"\nAverage Score: {sum(scores)/len(scores):.2f}")
    print(f"Min Score: {min(scores)}")
    print(f"Max Score: {max(scores)}")

if __name__ == "__main__":
    verify()
