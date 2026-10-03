"""Train the HiT-MAC executor on MATE with asynchronous A3C updates.

The rollout/update loop follows the original reference more closely:
local weights are refreshed per rollout, policy advantages use GAE, and
the critic uses bootstrapped discounted-return targets.
"""
from __future__ import annotations

import argparse
import os
import time

import gymnasium as gym
import numpy as np
import torch
import torch.multiprocessing as mp

import mate
from mate.algos.executor import (
    ExecutorConfig,
    ExecutorNet,
    build_target_features,
    generate_pseudo_goals,
    goal_conditioned_reward,
)
from mate.algos.shared import SharedAdam

def make_env(config_path: str | None):
    base = (
        gym.make("MultiAgentTracking-v0", config=config_path)
        if config_path
        else gym.make("MultiAgentTracking-v0")
    )
    return mate.MultiCamera.make(base, target_agent=mate.GreedyTargetAgent())

def flatten_camera_features(obs, goals, num_targets, device):
    features, masks = [], []
    for observation, goal in zip(obs, goals):
        feature, mask = build_target_features(observation, goal, num_targets)
        features.append(feature)
        masks.append(mask)
    return (
        torch.from_numpy(np.asarray(features)).float().to(device),
        torch.from_numpy(np.asarray(masks)).bool().to(device),
    )

def action_bounds(obs, rotation_only, device):
    from mate import constants as consts

    lows, highs = [], []
    for observation in obs:
        slices = consts.camera_observation_slices_of(
            int(round(observation[0])),
            int(round(observation[1])),
            int(round(observation[2])),
        )
        state = observation[slices["self_state"]]
        lows.append([
            -float(state[7]),
            0.0 if rotation_only else -float(state[8]),
        ])
        highs.append([
            float(state[7]),
            0.0 if rotation_only else float(state[8]),
        ])
    return (
        torch.tensor(lows, dtype=torch.float32, device=device),
        torch.tensor(highs, dtype=torch.float32, device=device),
    )

def test(model, args, num_episodes, seed_offset=0):
    """Evaluate a snapshot and report HiT-MAC-style test statistics."""
    env = make_env(args.config)
    num_targets = env.unwrapped.num_targets
    model.eval()
    episode_rewards = []
    episode_lengths = []
    episode_fps = []
    episode_rotations = []
    start_time = time.time()

    try:
        for episode in range(num_episodes):
            obs, _ = env.reset(
                seed=args.seed + 900000000 + seed_offset + episode
            )
            goals = generate_pseudo_goals(obs, num_targets)
            reward_sum = 0.0
            rotation_sum = 0.0
            steps = 0
            t0 = time.time()

            for step in range(args.max_steps):
                if step > 0 and step % args.goal_period == 0:
                    goals = generate_pseudo_goals(obs, num_targets)

                x, mask = flatten_camera_features(
                    obs, goals, num_targets, "cpu"
                )
                low, high = action_bounds(
                    obs, args.rotation_only, "cpu"
                )
                with torch.no_grad():
                    action, _, _, _ = model.act(
                        x, mask, low, high, deterministic=True
                    )

                action_np = action.cpu().numpy()
                if args.rotation_only:
                    action_np[:, 1] = 0.0

                next_obs, _, terminated, truncated, _ = env.step(action_np)
                reward = np.mean([
                    goal_conditioned_reward(
                        o, no, goal, act, args.reward_beta
                    )
                    for o, no, goal, act in zip(
                        obs, next_obs, goals, action_np
                    )
                ])
                reward_sum += float(reward)
                rotation_sum += float(np.mean(np.abs(action_np[:, 0])))
                steps += 1
                obs = next_obs

                if bool(terminated) or bool(truncated):
                    break

            elapsed = max(time.time() - t0, 1e-9)
            episode_rewards.append(reward_sum)
            episode_lengths.append(steps)
            episode_rotations.append(rotation_sum)
            episode_fps.append(steps / elapsed)

    finally:
        env.close()
        model.train()

    reward_array = np.asarray(episode_rewards, dtype=np.float64)
    length_array = np.asarray(episode_lengths, dtype=np.float64)
    rotation_array = np.asarray(episode_rotations, dtype=np.float64)
    fps_array = np.asarray(episode_fps, dtype=np.float64)

    # AG follows the reference test.py convention: reward accumulated per
    # unit of camera rotation. Guard the zero-rotation case.
    ag_per_episode = np.divide(
        reward_array,
        rotation_array,
        out=np.zeros_like(reward_array),
        where=rotation_array > 1e-8,
    )

    return {
        "elapsed": time.time() - start_time,
        "ave_eps_reward": float(reward_array.mean()),
        "ave_eps_length": float(length_array.mean()),
        "reward_step": float(reward_array.sum() / max(length_array.sum(), 1.0)),
        "fps": float(fps_array.mean()),
        "mean_reward": float(reward_array.mean()),
        "std_reward": float(reward_array.std()),
        "AG": float(ag_per_episode.mean()),
    }

def train(
    rollout,
    bootstrap,
    local_model,
    shared_model,
    optimizer,
    optimizer_lock,
    cfg,
):
    """Apply one A3C update from a rollout.

    Each transition stores a scalar reward averaged across cameras. Expand it
    to the critic's per-camera shape, preserving the existing reward design.
    """
    if not rollout:
        return

    rewards = [
        torch.full_like(item[6], float(item[7]))
        for item in rollout
    ]
    values = [item[6] for item in rollout]
    log_probs = [item[4] for item in rollout]
    entropies = [item[5] for item in rollout]

    # Discounted-return targets for the critic.
    returns = []
    running_return = bootstrap.detach()
    for reward in reversed(rewards):
        running_return = reward + cfg.gamma * running_return
        returns.append(running_return)
    returns.reverse()

    # GAE(lambda) for the policy; tau is lambda in this recurrence.
    gae = torch.zeros_like(bootstrap)
    next_value = bootstrap.detach()
    policy_loss = torch.zeros((), device=bootstrap.device)
    value_loss = torch.zeros((), device=bootstrap.device)

    for t in reversed(range(len(rollout))):
        delta = rewards[t] + cfg.gamma * next_value - values[t]
        gae = delta + cfg.gamma * cfg.tau * gae
        policy_loss = policy_loss - (log_probs[t] * gae.detach()).mean()
        policy_loss = policy_loss - cfg.entropy_coef * entropies[t].mean()
        value_loss = value_loss + 0.5 * (returns[t] - values[t]).pow(2).mean()
        next_value = values[t].detach()

    total_loss = policy_loss + value_loss
    local_model.zero_grad(set_to_none=True)
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(local_model.parameters(), cfg.grad_clip)

    # Keep gradient transfer and the shared optimizer step atomic across workers.
    with optimizer_lock:
        optimizer.zero_grad()
        for local_param, shared_param in zip(
            local_model.parameters(), shared_model.parameters()
        ):
            shared_param.grad = (
                local_param.grad.detach().clone()
                if local_param.grad is not None
                else None
            )
        optimizer.step()

    rollout.clear()

def worker(
    rank,
    args,
    shared_model,
    optimizer,
    episode_counter,
    episode_lock,
    optimizer_lock,
    best_score,
    best_lock,
):
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.set_num_threads(1)

    device = torch.device("cpu")
    env = make_env(args.config)
    num_targets = env.unwrapped.num_targets
    cfg = ExecutorConfig(
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        tau=args.tau,
        entropy_coef=args.entropy,
        learning_rate=args.lr,
        grad_clip=args.grad_clip,
        rollout_steps=args.update_frequency,
        goal_period=args.goal_period,
        reward_beta=args.reward_beta,
        rotation_only=args.rotation_only,
    )
    local_model = ExecutorNet(5, cfg.hidden_dim, action_dim=2).to(device)
    local_model.train()

    while True:
        with episode_lock:
            if episode_counter.value >= args.episodes:
                break
            episode_counter.value += 1
            episode = episode_counter.value

        obs, _ = env.reset(seed=args.seed + rank * 100000 + episode)
        goals = generate_pseudo_goals(obs, num_targets)
        rollout = []
        ep_return = 0.0

        # Load once at the beginning of the first rollout.
        with optimizer_lock:
            local_model.load_state_dict(shared_model.state_dict())

        for step in range(args.max_steps):
            # Refresh pseudo-goals at the configured interval.
            if step > 0 and step % cfg.goal_period == 0:
                goals = generate_pseudo_goals(obs, num_targets)

            x, mask = flatten_camera_features(obs, goals, num_targets, device)
            low, high = action_bounds(obs, cfg.rotation_only, device)
            action, logp, entropy, value = local_model.act(x, mask, low, high)
            action_np = action.detach().cpu().numpy()

            if cfg.rotation_only:
                action_np[:, 1] = 0.0

            next_obs, _, terminated, truncated, _ = env.step(action_np)
            terminated = bool(terminated)
            truncated = bool(truncated)
            done = terminated or truncated

            rewards = np.asarray([
                goal_conditioned_reward(o, no, goal, act, cfg.reward_beta)
                for o, no, goal, act in zip(obs, next_obs, goals, action_np)
            ], dtype=np.float32)
            reward = float(rewards.mean())
            ep_return += reward

            rollout.append((x, mask, low, high, logp, entropy, value, reward))
            obs = next_obs

            rollout_end = len(rollout) >= cfg.rollout_steps or done
            if rollout_end:
                # True termination has no future value. A time-limit
                # truncation and an ordinary rollout boundary are bootstrapped.
                if terminated:
                    bootstrap = torch.zeros_like(value).detach()
                else:
                    with torch.no_grad():
                        bx, bm = flatten_camera_features(
                            obs, goals, num_targets, device
                        )
                        _, _, bootstrap = local_model(bx, bm)

                train(
                    rollout,
                    bootstrap,
                    local_model,
                    shared_model,
                    optimizer,
                    optimizer_lock,
                    cfg,
                )

                # Match the reference: refresh local parameters for the next
                # rollout/update cycle, not merely at episode boundaries.
                if not done:
                    with optimizer_lock:
                        local_model.load_state_dict(shared_model.state_dict())

            if done:
                break

        if rank == 0:
            eval_model = ExecutorNet(5, args.hidden_dim, action_dim=2)
            with optimizer_lock:
                eval_model.load_state_dict(shared_model.state_dict())
            metrics = test(
                eval_model, args, args.eval_episodes, episode
            )
            eval_mean = metrics["ave_eps_reward"]
            improved = False
            with best_lock:
                if eval_mean > best_score.value:
                    best_score.value = eval_mean
                    improved = True

            print(
                "Time {0}, ave eps reward {1}, ave eps length {2}, "
                "reward step {3}, FPS {4}, mean reward {5}, "
                "std reward {6}, AG {7}{8}".format(
                    time.strftime(
                        "%Hh %Mm %Ss",
                        time.gmtime(metrics["elapsed"]),
                    ),
                    np.around(metrics["ave_eps_reward"], 2),
                    np.around(metrics["ave_eps_length"], 2),
                    np.around(metrics["reward_step"], 2),
                    np.around(metrics["fps"], 2),
                    np.around(metrics["mean_reward"], 2),
                    np.around(metrics["std_reward"], 2),
                    np.around(metrics["AG"], 2),
                    " [best]" if improved else "",
                ),
                flush=True,
            )

            if improved:
                os.makedirs(
                    os.path.dirname(args.best_save) or ".", exist_ok=True
                )
                torch.save(
                    {
                        "model": eval_model.state_dict(),
                        "config": vars(args),
                        "eval_metrics": metrics,
                        "evaluation_episode": episode,
                    },
                    args.best_save,
                )

    env.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="mate/assets/MATE-4v5-0.yaml")
    parser.add_argument("--episodes", type=int, default=50000)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--update-frequency", type=int, default=20)
    parser.add_argument("--goal-period", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--grad-clip", type=float, default=50.0)
    parser.add_argument("--reward-beta", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--best-save", default="trainedModel/executor_best.pth")
    parser.add_argument("--rotation-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save", default="trainedModel/executor.pth")
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    mp.set_start_method("spawn", force=True)

    shared_model = ExecutorNet(5, args.hidden_dim, action_dim=2)
    shared_model.share_memory()
    optimizer = SharedAdam(shared_model.parameters(), lr=args.lr)

    episode_counter = mp.Value("i", 0)
    episode_lock = mp.Lock()
    optimizer_lock = mp.Lock()
    best_lock = mp.Lock()
    best_score = mp.Value("d", float("-inf"))
    processes = []
    for rank in range(args.workers):
        process = mp.Process(
            target=worker,
            args=(
                rank, args, shared_model, optimizer, episode_counter,
                episode_lock, optimizer_lock, best_score, best_lock,
            ),
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(
                f"Training worker exited with code {process.exitcode}"
            )

    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    torch.save(
        {"model": shared_model.state_dict(), "config": vars(args)},
        args.save,
    )
    print(f"saved executor -> {args.save}")

if __name__ == "__main__":
    main()
