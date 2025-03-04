import gymnasium as gym
from wrapper import RandomActionWrapper

if __name__ == "__main__":
    env = RandomActionWrapper(gym.make("CartPole-v1"))

    obs = env.reset()
    total_reward = 0.0

    while True:
        obs, reward, done, _, _ = env.step(0)
        total_reward += reward
        if done:
            break

    print(f"Reward got: {total_reward:.2f}")