import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import src.envs
from src.envs.bloxorz_env import BloxorzEnv
from src.utils.block import Orientation
from src.utils.renderer import render_frame

_ACTION_NAMES = {0: "N(up)", 1: "E(right)", 2: "S(down)", 3: "W(left)"}
_ORIENT_NAMES = {
    Orientation.STAND:  "STAND",
    Orientation.FLAT_V: "FLAT_V",
    Orientation.FLAT_H: "FLAT_H",
}
HUD_H = 64
BG_HUD = (25, 25, 35)
FG_INFO = (210, 210, 210)
FG_DIM  = (120, 120, 130)
FG_WIN  = (80,  220, 80)
FG_LOSE = (220, 80,  80)


def _draw_hud(screen, font, env, steps, last_action, message, msg_color, level):
    import pygame
    frame_h = env._grid.shape[0] * 32
    frame_w = env._grid.shape[1] * 32
    r = pygame.Rect(0, frame_h, frame_w, HUD_H)
    pygame.draw.rect(screen, BG_HUD, r)

    block = env._block
    pos_str    = f"({block.r}, {block.c})" if block else "?"
    orient_str = _ORIENT_NAMES.get(block.orientation, "?") if block else "?"
    act_str    = _ACTION_NAMES.get(last_action, "—")
    bridges    = env._bridge_states

    if message:
        l1 = font.render(message, True, msg_color)
        l2 = font.render("R = restart   Q = quit", True, FG_DIM)
    else:
        info = (
            f"L{level}  step={steps}  pos={pos_str}  {orient_str}"
            f"  last={act_str}"
        )
        if bridges:
            info += f"  bridges={[int(b) for b in bridges]}"
        l1 = font.render(info, True, FG_INFO)
        l2 = font.render("↑↓←→ (or WASD) = move   R = restart   Q = quit", True, FG_DIM)

    screen.blit(l1, (8, frame_h + 8))
    screen.blit(l2, (8, frame_h + 36))


def parse_args():
    p = argparse.ArgumentParser(description="Play Bloxorz manually")
    p.add_argument("--level", type=int, default=1, help="Level number 1-33")
    p.add_argument("--fps",   type=int, default=60)
    return p.parse_args()


def main():
    args = parse_args()
    import pygame

    env = BloxorzEnv(level=args.level, render_mode=None)
    obs, _ = env.reset()

    pygame.init()
    font = pygame.font.SysFont("monospace", 14)

    frame = render_frame(env)
    h_px, w_px = frame.shape[:2]
    screen = pygame.display.set_mode((w_px, h_px + HUD_H))
    pygame.display.set_caption(f"Bloxorz — Level {args.level}")
    clock = pygame.time.Clock()

    steps       = 0
    last_action = None
    message     = ""
    msg_color   = FG_WIN
    done        = False

    while True:
        action = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close(); pygame.quit(); return
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    env.close(); pygame.quit(); return
                if event.key == pygame.K_r:
                    obs, _ = env.reset()
                    steps = 0; last_action = None
                    message = ""; done = False
                if not done:
                    if   event.key in (pygame.K_UP,    pygame.K_w): action = 0
                    elif event.key in (pygame.K_RIGHT,  pygame.K_d): action = 1
                    elif event.key in (pygame.K_DOWN,   pygame.K_s): action = 2
                    elif event.key in (pygame.K_LEFT,   pygame.K_a): action = 3

        if action is not None:
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            last_action = action
            if info.get("win"):
                message   = f"YOU WIN! — {steps} steps"
                msg_color = FG_WIN
                done = True
            elif info.get("fall"):
                message   = f"FELL on step {steps}"
                msg_color = FG_LOSE
                done = True
            elif truncated:
                message   = f"TIMEOUT at {steps} steps"
                msg_color = FG_LOSE
                done = True

        frame = render_frame(env)
        surf = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
        screen.blit(surf, (0, 0))
        _draw_hud(screen, font, env, steps, last_action, message, msg_color, args.level)
        pygame.display.flip()
        clock.tick(args.fps)


if __name__ == "__main__":
    main()
