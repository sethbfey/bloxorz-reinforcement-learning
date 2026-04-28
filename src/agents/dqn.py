import argparse
import copy
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import sys
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import src.envs
from src.envs.bloxorz_env import BloxorzEnv, MAX_H, MAX_W

BUFFER_SIZE        = 50_000
BATCH_SIZE         = 64
LR                 = 1e-3
GAMMA              = 0.99
EPS_START          = 1.0
EPS_END            = 0.10
EPS_DECAY_STEPS    = 500_000
LEARNING_STARTS    = 1_000
TRAIN_FREQ         = 4
TARGET_UPDATE_FREQ = 2_000
GRAD_CLIP          = 10.0
WIN_REPEAT         = 50
TRAJ_REPEAT        = 3
WIN_MC_REPEAT      = 3


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--level",           type=int,  default=1)
    p.add_argument("--random-start",    action="store_true")
    p.add_argument("--fixed-start-frac", type=float, default=0.3)
    p.add_argument("--start-temp",       type=float, default=0.3)
    p.add_argument("--curriculum-start", type=int,   default=None)
    p.add_argument("--curriculum-steps", type=int,   default=40_000)
    p.add_argument("--seed",            type=int,  default=1)
    p.add_argument("--total-timesteps", type=int,  default=500_000)
    p.add_argument("--eval-interval",   type=int,  default=10_000)
    p.add_argument("--eval-episodes",   type=int,  default=200)
    p.add_argument("--potential-shaping", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-model",      action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--track",           action="store_true")
    p.add_argument("--wandb-project",   type=str,  default="bloxorz-rl")
    return p.parse_args()


class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape: tuple) -> None:
        self._obs      = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._actions  = np.zeros(capacity, dtype=np.int64)
        self._rewards  = np.zeros(capacity, dtype=np.float32)
        self._terms    = np.zeros(capacity, dtype=np.float32)
        self._capacity = capacity
        self._ptr      = 0
        self._size     = 0

    def add(self, obs, action: int, reward: float, next_obs, terminated: bool) -> None:
        self._obs[self._ptr]      = obs
        self._next_obs[self._ptr] = next_obs
        self._actions[self._ptr]  = action
        self._rewards[self._ptr]  = reward
        self._terms[self._ptr]    = float(terminated)
        self._ptr  = (self._ptr + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample(self, batch_size: int, device: torch.device):
        idx = np.random.randint(0, self._size, size=batch_size)
        return (
            torch.as_tensor(self._obs[idx],      device=device),
            torch.as_tensor(self._actions[idx],  device=device),
            torch.as_tensor(self._rewards[idx],  device=device),
            torch.as_tensor(self._next_obs[idx], device=device),
            torch.as_tensor(self._terms[idx],    device=device),
        )

    def __len__(self) -> int:
        return self._size


class QNetwork(nn.Module):
    def __init__(self, in_channels: int = 3, n_actions: int = 4) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * MAX_H * MAX_W, 256), nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x).flatten(1))


def evaluate(q_net: QNetwork, env: BloxorzEnv, n_episodes: int) -> dict:
    eval_net = copy.deepcopy(q_net).cpu()
    eval_net.eval()
    cpu = torch.device("cpu")

    wins, total_ret, total_len = 0, 0.0, 0
    with torch.no_grad():
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done, ep_ret, ep_len = False, 0.0, 0
            while not done:
                q_vals = eval_net(torch.as_tensor(obs, dtype=torch.float32, device=cpu).unsqueeze(0))
                action = int(q_vals.argmax().item())
                obs, reward, terminated, truncated, info = env.step(action)
                ep_ret += reward
                ep_len += 1
                done = terminated or truncated
            wins      += int(info.get("win", False))
            total_ret += ep_ret
            total_len += ep_len
    return {
        "win_rate":    wins / n_episodes,
        "mean_return": total_ret / n_episodes,
        "mean_len":    total_len / n_episodes,
    }


def main():
    args     = parse_args()
    run_name = f"dqn_level{args.level}_seed{args.seed}_{int(time.time())}"

    if args.track:
        import wandb
        wandb.init(project=args.wandb_project, name=run_name,
                   config=vars(args), sync_tensorboard=True)

    writer = SummaryWriter(f"runs/{run_name}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    print(f"device={device}  run={run_name}", flush=True)

    env      = BloxorzEnv(level=args.level, random_start=args.random_start,
                          fixed_start_frac=args.fixed_start_frac,
                          start_temp=args.start_temp,
                          potential_shaping=args.potential_shaping)
    eval_env = BloxorzEnv(level=args.level, random_start=False)
    obs_shape   = env.observation_space.shape
    in_channels = obs_shape[0]
    n_actions   = int(env.action_space.n)
    if env._has_splits:
        print(f"  split level: using {in_channels}-channel obs, {n_actions}-action network", flush=True)

    buffer     = ReplayBuffer(BUFFER_SIZE, obs_shape)
    q_net      = QNetwork(in_channels, n_actions).to(device)
    target_net = QNetwork(in_channels, n_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()
    optimizer  = optim.Adam(q_net.parameters(), lr=LR)

    obs, _ = env.reset(seed=args.seed)
    # Share the already-computed valid_states with the eval env so it never recomputes them.
    eval_env._valid_states = env._valid_states

    rng = np.random.default_rng(args.seed)
    curr_states: list | None = None
    curriculum_max_dist: int | None = None
    if args.curriculum_start is not None:
        if env._dist_to_win is None:
            env._dist_to_win = env._compute_dist_to_win()
        curriculum_max_dist = args.curriculum_start
        curr_states = [s for s in env._valid_states if env._dist_to_win.get(s, 9999) <= curriculum_max_dist]
        print(f"  curriculum: d≤{curriculum_max_dist}, {len(curr_states)} states", flush=True)

    ep_ret, ep_len = 0.0, 0
    ep_traj:       list[tuple] = []
    best_win_rate  = 0.0
    best_mean_len  = float("inf")  # tiebreaker once win_rate ceiling is reached
    next_eval_at   = args.eval_interval
    start_time     = time.time()

    recent_returns: list[float] = []
    recent_lengths: list[int]   = []
    recent_wins:    list[int]   = []

    for step in range(1, args.total_timesteps + 1):

        eps = max(EPS_END, EPS_START + (EPS_END - EPS_START) * step / EPS_DECAY_STEPS)
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = q_net(
                    torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                ).argmax().item()

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        buffer.add(obs, action, reward, next_obs, terminated)
        ep_traj.append((obs, action, reward, next_obs, terminated))
        ep_ret += reward
        ep_len += 1

        if done:
            if info.get("win", False):
                t_obs, t_act, t_rew, t_nobs, t_term = ep_traj[-1]
                for _ in range(WIN_REPEAT - 1):
                    buffer.add(t_obs, t_act, t_rew, t_nobs, t_term)
                for t_obs, t_act, t_rew, t_nobs, t_term in ep_traj[:-1]:
                    for _ in range(TRAJ_REPEAT):
                        buffer.add(t_obs, t_act, t_rew, t_nobs, t_term)
                if WIN_MC_REPEAT > 0:
                    G = 0.0
                    for t_obs, t_act, t_rew, t_nobs, _ in reversed(ep_traj):
                        G = t_rew + GAMMA * G
                        for _ in range(WIN_MC_REPEAT):
                            buffer.add(t_obs, t_act, G, t_nobs, True)
            recent_returns.append(ep_ret)
            recent_lengths.append(ep_len)
            recent_wins.append(int(info.get("win", False)))
            ep_ret, ep_len = 0.0, 0
            ep_traj = []
            if curr_states is not None:
                state = curr_states[rng.integers(len(curr_states))]
                obs, _ = env.reset(options={"start_state": state})
            else:
                obs, _ = env.reset()
        else:
            obs = next_obs

        if step >= LEARNING_STARTS and step % TRAIN_FREQ == 0 and len(buffer) >= BATCH_SIZE:
            b_obs, b_act, b_rew, b_next_obs, b_term = buffer.sample(BATCH_SIZE, device)

            with torch.no_grad():
                next_actions = q_net(b_next_obs).argmax(1)
                next_q = target_net(b_next_obs).gather(
                    1, next_actions.unsqueeze(1)
                ).squeeze(1)
                target_q = b_rew + GAMMA * next_q * (1.0 - b_term)

            current_q = q_net(b_obs).gather(1, b_act.unsqueeze(1)).squeeze(1)
            loss = nn.functional.smooth_l1_loss(current_q, target_q)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), GRAD_CLIP)
            optimizer.step()

            writer.add_scalar("losses/td_loss", loss.item(), step)

        if step % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(q_net.state_dict())

        if curriculum_max_dist is not None and step % args.curriculum_steps == 0:
            new_dist = curriculum_max_dist + 1
            new_states = [s for s in env._valid_states if env._dist_to_win.get(s, 9999) <= new_dist]
            if len(new_states) > len(curr_states):
                curriculum_max_dist = new_dist
                curr_states = new_states
                writer.add_scalar("charts/curriculum_max_dist", curriculum_max_dist, step)
                print(f"  curriculum expanded: d≤{curriculum_max_dist}, {len(curr_states)} states", flush=True)

        if step % 1_000 == 0 and recent_returns:
            sps = int(step / (time.time() - start_time))
            writer.add_scalar("charts/epsilon",          eps,                             step)
            writer.add_scalar("charts/steps_per_second", sps,                             step)
            writer.add_scalar("charts/ep_return",        np.mean(recent_returns[-100:]),  step)
            writer.add_scalar("charts/ep_length",        np.mean(recent_lengths[-100:]),  step)
            writer.add_scalar("charts/train_win_rate",   np.mean(recent_wins[-100:]),     step)
            if curriculum_max_dist is not None:
                writer.add_scalar("charts/curriculum_max_dist", curriculum_max_dist,      step)

        if step >= next_eval_at:
            s       = evaluate(q_net, eval_env, args.eval_episodes)
            # Best = strictly higher win_rate, OR equal win_rate with strictly shorter mean_len.
            is_best = (
                s["win_rate"] > best_win_rate
                or (s["win_rate"] == best_win_rate and s["mean_len"] < best_mean_len)
            )
            writer.add_scalar("eval/win_rate",    s["win_rate"],    step)
            writer.add_scalar("eval/mean_return", s["mean_return"], step)
            writer.add_scalar("eval/mean_len",    s["mean_len"],    step)
            sps = int(step / (time.time() - start_time))
            train_wins = sum(recent_wins[-500:])
            print(f"  step={step:>9,}  eps={eps:.3f}  win_rate={s['win_rate']:>5.1%}"
                  f"  train_wins={train_wins:>3d}  mean_len={s['mean_len']:>5.1f}  sps={sps:,}"
                  + ("  ★ best" if is_best else ""), flush=True)
            if is_best and args.save_model:
                best_win_rate = s["win_rate"]
                best_mean_len = s["mean_len"]
                os.makedirs("runs", exist_ok=True)
                torch.save({
                    "q_network":   q_net.state_dict(),
                    "in_channels": in_channels,
                    "n_actions":   n_actions,
                    "args":        vars(args),
                }, f"runs/{run_name}_best.pt")
            next_eval_at += args.eval_interval

    final = evaluate(q_net, eval_env, args.eval_episodes)
    print(f"\nFinal | level={args.level}  win_rate={final['win_rate']:.1%}"
          f"  mean_len={final['mean_len']:.1f}  mean_return={final['mean_return']:.2f}", flush=True)
    writer.add_scalar("eval/win_rate",    final["win_rate"],    step)
    writer.add_scalar("eval/mean_return", final["mean_return"], step)
    writer.add_scalar("eval/mean_len",    final["mean_len"],    step)

    if args.save_model:
        os.makedirs("runs", exist_ok=True)
        path = f"runs/{run_name}.pt"
        torch.save({
            "q_network":   q_net.state_dict(),
            "in_channels": in_channels,
            "n_actions":   n_actions,
            "args":        vars(args),
        }, path)
        print(f"Saved → {path}")

    env.close()
    eval_env.close()
    writer.close()
    if args.track:
        wandb.finish()


if __name__ == "__main__":
    main()
