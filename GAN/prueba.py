import gymnasium as gym
import ale_py

prueba = gym.make("ALE/Breakout-v5", render_mode="human")
print(prueba)