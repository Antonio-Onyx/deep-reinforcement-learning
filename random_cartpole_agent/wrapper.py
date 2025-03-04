import gymnasium as gym
import random

class RandomActionWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env, epsilon: float=0.1):
        super(RandomActionWrapper, self).__init__(env)
        self.epsilon = epsilon

    def action(self, action: gym.core.WrapperActType) -> gym.core.WrapperActType:
        if random.random() < self.epsilon:
            action = self.env.action_space.sample()
            print(f"Random action {action}")
        return action