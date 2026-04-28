import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import src.envs
from src.envs.bloxorz_env import BloxorzEnv
from src.agents.dqn import QNetwork


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--level", type=int, default=1)
    p.add_argument("--fps",   type=int, default=4)
    p.add_argument("--episodes", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    q_net = QNetwork()
    q_net.load_state_dict(ckpt["q_network"])
    q_net.eval()

    env = BloxorzEnv(level=args.level, render_mode="human")

    import pygame
    pygame.init()

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
            with torch.no_grad():
                action = q_net(
                    torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                ).argmax().item()
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            step += 1
            done = terminated or truncated
            time.sleep(1.0 / args.fps)
        result = "WIN" if info.get("win") else ("FALL" if info.get("fall") else "TIMEOUT")
        print(f"Episode {ep+1}: {result} in {step} steps")

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
