from atari_gan import *
import ale_py

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", default="cpu", help="Device name, default=cpu")
    args = parser.parse_args()


    #print(args)
    device = torch.device(args.dev)
    envs = [
        InputWrapper(gym.make(name))
        for name in ('ALE/Breakout-v5', 'ALE/AirRaid-v5', 'ALE/Pong-v5')
    ]
    shape = envs[0].observation_space.shape

    print(f"Device: {device}")
    print(f"shape of observations space: {shape}")