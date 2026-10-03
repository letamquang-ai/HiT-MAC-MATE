"""Train the HiT-MAC coordinator on MATE with an on-policy actor-critic loop.

The coordinator selects camera-target goals; a pretrained executor translates
each camera's selected goals into continuous camera controls. This script uses
the environment team reward and bootstrapped returns to update the coordinator.
"""
from __future__ import annotations

import argparse
import os
import random

import gymnasium as gym
import numpy as np
import torch
from torch import nn

import mate
from mate.algos.coordinator import CoordinatorNet
from mate.algos.executor import build_target_features


def make_env(config):
    base = gym.make("MultiAgentTracking-v0", config=config) if config else gym.make("MultiAgentTracking-v0")
    return mate.MultiCamera.make(base, target_agent=mate.GreedyTargetAgent())


def coordinator_inputs(observations, num_targets, device):
    """Encode each camera's currently sensed targets as [1,C,T,5]."""
    features, masks = [], []
    all_goals = np.ones(num_targets, dtype=np.float32)
    for obs in observations:
        f, m = build_target_features(obs, all_goals, num_targets)
        features.append(f)
        masks.append(m)
    x = torch.as_tensor(np.asarray(features), dtype=torch.float32, device=device).unsqueeze(0)
    mask = torch.as_tensor(np.asarray(masks), dtype=torch.bool, device=device).unsqueeze(0)
    return x, mask


def scripted_executor(observation, assigned_goals, num_targets, rotation_only=True):
    """Section-B-style scripted camera controller: turn toward an assigned visible target."""
    from mate import constants as consts
    obs = np.asarray(observation, dtype=np.float64)
    slices = consts.camera_observation_slices_of(
        int(round(obs[0])), int(round(obs[1])), int(round(obs[2]))
    )
    state = obs[slices["self_state"]]
    targets = obs[slices["opponent_states_with_mask"]].reshape(
        num_targets, consts.TARGET_STATE_DIM_PUBLIC + 1
    )
    candidates = [
        j for j in range(num_targets)
        if assigned_goals[j] > 0.5 and targets[j, consts.TARGET_STATE_DIM_PUBLIC] > 0.5
    ]
    if not candidates:
        return np.zeros(2, dtype=np.float32)
    # Choose the closest assigned target, then rotate toward its bearing.
    xy = state[:2]
    j = min(candidates, key=lambda k: np.linalg.norm(targets[k, :2] - xy))
    delta = targets[j, :2] - xy
    bearing = np.degrees(np.arctan2(delta[1], delta[0]))
    heading = np.degrees(np.arctan2(state[4], state[3]))
    error = ((bearing - heading + 180.0) % 360.0) - 180.0
    max_rotation = float(state[7])
    rotation = float(np.clip(error, -max_rotation, max_rotation))
    return np.asarray([rotation, 0.0], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="mate/assets/MATE-4v5-0.yaml")
    parser.add_argument("--episodes", type=int, default=50000)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--update-frequency", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--attention-dim", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--grad-clip", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--rotation-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save", default="trainedModel/coordinator.pth")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(args.config)
    num_cameras = env.unwrapped.num_cameras
    num_targets = env.unwrapped.num_targets
    coordinator = CoordinatorNet(5, args.hidden_dim, args.attention_dim).to(device)
    optimizer = torch.optim.Adam(coordinator.parameters(), lr=args.lr)

    episode_returns = []
    for episode in range(1, args.episodes + 1):
        obs, _ = env.reset(seed=args.seed + episode)
        log_probs, entropies, values, rewards = [], [], [], []
        ep_return = 0.0
        for step in range(args.max_steps):
            x, mask = coordinator_inputs(obs, num_targets, device)
            goals, logp, entropy, value = coordinator.act(x, mask, sample=True)
            goal_np = goals[0].detach().cpu().numpy()
            actions = np.asarray([
                scripted_executor(camera_obs, goal, num_targets, args.rotation_only)
                for camera_obs, goal in zip(obs, goal_np)
            ], dtype=np.float32)
            next_obs, env_reward, terminated, truncated, _ = env.step(actions)
            done = bool(terminated or truncated)
            reward = float(np.asarray(env_reward, dtype=np.float32).mean())
            log_probs.append(logp.squeeze(0))
            entropies.append(entropy.squeeze(0))
            values.append(value.squeeze(0))
            rewards.append(reward)
            ep_return += reward
            obs = next_obs

            if done or len(rewards) >= args.update_frequency:
                with torch.no_grad():
                    if done:
                        bootstrap = torch.zeros((), device=device)
                    else:
                        bx, bm = coordinator_inputs(obs, num_targets, device)
                        _, bootstrap = coordinator(bx, bm)
                        bootstrap = bootstrap.squeeze(0)
                returns = []
                running = bootstrap
                for r in reversed(rewards):
                    running = torch.as_tensor(r, dtype=torch.float32, device=device) + args.gamma * running
                    returns.append(running)
                returns.reverse()
                returns_t = torch.stack(returns)
                values_t = torch.stack(values)
                logp_t = torch.stack(log_probs)
                entropy_t = torch.stack(entropies)
                advantage = returns_t - values_t
                policy_loss = -(logp_t * advantage.detach()).mean()
                value_loss = advantage.pow(2).mean()
                loss = policy_loss + args.value_coef * value_loss - args.entropy * entropy_t.mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(coordinator.parameters(), args.grad_clip)
                optimizer.step()
                log_probs.clear(); entropies.clear(); values.clear(); rewards.clear()
            if done:
                break
        episode_returns.append(ep_return)
        if episode % args.log_interval == 0:
            mean_return = float(np.mean(episode_returns[-args.log_interval:]))
            print(f"episode={episode:6d} return={ep_return:8.3f} mean={mean_return:8.3f}", flush=True)

    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    torch.save({"model": coordinator.state_dict(), "config": vars(args)}, args.save)
    print(f"saved coordinator -> {args.save}")
    env.close()


if __name__ == "__main__":
    main()
