from snake_env import SnakeEnv
import random
import time

def run_random_episodes(num_episodes=30, seed=0):
    random.seed(seed)
    env = SnakeEnv(width=10, height=10, step_cost=-0.01, seed=seed)

    for episode in range(num_episodes):
        obs = env.reset(seed=seed + episode)  # vary seeds per episode for diversity
        done = False
        total_reward = 0

        print(f"\n=== Episode {episode + 1} ===")
        while not done:
            action = random.choice([0, 1, 2, 3])
            obs, reward, done, info = env.step(action)
            total_reward += reward

            env.render()
            print(f"Step reward: {reward}  Total reward: {total_reward}  Score: {env.score}\n")
            time.sleep(0.1)

        print(f"Episode finished with score {env.score} and total reward {total_reward}")

if __name__ == "__main__":
    run_random_episodes()
