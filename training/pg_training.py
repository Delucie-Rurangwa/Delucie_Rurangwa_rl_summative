"""Policy-gradient methods on the battery-swap station, built with TorchRL:

  * REINFORCE  - Monte-Carlo policy gradient on complete days, optional return normalisation
                 as a baseline, entropy bonus.
  * PPO        - clipped surrogate objective, GAE advantages, several epochs of
                 mini-batch updates per batch of experience, value network.
  * A2C        - advantage actor-critic: one synchronous update per short rollout,
                 GAE / n-step advantages, value network.

    uv run python -m training.pg_training --algo ppo            # all 10 PPO runs
    uv run python -m training.pg_training --algo a2c --run 4    # one run
    uv run python -m training.pg_training --algo all            # REINFORCE, PPO and A2C
"""

from __future__ import annotations

import argparse

import torch
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.modules import OneHotCategorical, ProbabilisticActor, ValueOperator
from torchrl.objectives import A2CLoss, ClipPPOLoss
from torchrl.objectives.value import GAE

try:
    from torchrl.collectors import Collector
except ImportError:  # pragma: no cover
    from torchrl.collectors import SyncDataCollector as Collector

from training.common import (N_ACTIONS, RunLogger, evaluate, finished_episode_rewards, make_env, mlp,
                             save_model, set_seed, write_sweep_table)

TOTAL_FRAMES = 150_000      # same experience budget as DQN
DAY = 144                   # steps in one episode

# --------------------------------------------------------------------------- sweeps
REINFORCE_BASE = dict(lr=1e-3, gamma=0.99, episodes_per_batch=4, entropy_coef=0.01,
                      normalize_returns=True, hidden=64, activation="tanh")
REINFORCE_SWEEP = [
    {},                                              # 1 baseline
    dict(lr=3e-4),                                   # 2 slower learning rate
    dict(lr=3e-3),                                   # 3 faster learning rate
    dict(gamma=0.95),                                # 4 shorter horizon
    dict(gamma=0.999),                               # 5 almost undiscounted
    dict(episodes_per_batch=1),                      # 6 one day per update (high variance)
    dict(episodes_per_batch=8),                      # 7 eight days per update
    dict(entropy_coef=0.0),                          # 8 no entropy bonus
    dict(entropy_coef=0.05),                         # 9 strong entropy bonus
    dict(normalize_returns=False, hidden=128),       # 10 no baseline, bigger net
]
REINFORCE_COLUMNS = ["lr", "gamma", "episodes_per_batch", "entropy_coef", "normalize_returns", "hidden"]

PPO_BASE = dict(lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2, entropy_coef=0.01,
                critic_coef=0.5, epochs=4, frames_per_batch=1440, minibatch=240,
                hidden=64, activation="tanh")
PPO_SWEEP = [
    {},                                              # 1 baseline
    dict(lr=1e-4),                                   # 2 slower learning rate
    dict(lr=1e-3),                                   # 3 faster learning rate
    dict(gamma=0.95),                                # 4 shorter horizon
    dict(gae_lambda=0.8),                            # 5 more bias, less variance in advantages
    dict(clip_eps=0.1),                              # 6 tight clipping
    dict(clip_eps=0.3),                              # 7 loose clipping
    dict(entropy_coef=0.0),                          # 8 no entropy bonus
    dict(epochs=10, minibatch=144),                  # 9 many updates per batch
    dict(frames_per_batch=576, hidden=128),          # 10 short rollouts, bigger net
]
PPO_COLUMNS = ["lr", "gamma", "gae_lambda", "clip_eps", "entropy_coef", "epochs",
               "frames_per_batch", "minibatch", "hidden"]

A2C_BASE = dict(lr=7e-4, gamma=0.99, gae_lambda=1.0, n_steps=32, entropy_coef=0.01,
                critic_coef=0.5, hidden=64, activation="tanh", max_grad_norm=0.5)
A2C_SWEEP = [
    {},                                              # 1 baseline (n-step returns)
    dict(lr=2e-4),                                   # 2 slower learning rate
    dict(lr=2e-3),                                   # 3 faster learning rate
    dict(gamma=0.95),                                # 4 shorter horizon
    dict(gae_lambda=0.95),                           # 5 GAE instead of n-step
    dict(n_steps=8),                                 # 6 very short rollouts
    dict(n_steps=144),                               # 7 one full day per update
    dict(entropy_coef=0.0),                          # 8 no entropy bonus
    dict(entropy_coef=0.05),                         # 9 strong entropy bonus
    dict(critic_coef=1.0, hidden=128),               # 10 heavier critic, bigger net
]
A2C_COLUMNS = ["lr", "gamma", "gae_lambda", "n_steps", "entropy_coef", "critic_coef", "hidden"]

ALGOS = {
    "reinforce": (REINFORCE_BASE, REINFORCE_SWEEP, REINFORCE_COLUMNS),
    "ppo": (PPO_BASE, PPO_SWEEP, PPO_COLUMNS),
    "a2c": (A2C_BASE, A2C_SWEEP, A2C_COLUMNS),
}


# --------------------------------------------------------------------------- networks
def make_actor(env, cfg):
    act = nn.ReLU if cfg["activation"] == "relu" else nn.Tanh
    logits_net = mlp(N_ACTIONS, cfg["hidden"], activation=act)
    module = TensorDictModule(logits_net, in_keys=["observation"], out_keys=["logits"])
    actor = ProbabilisticActor(module, spec=env.action_spec, in_keys=["logits"],
                               distribution_class=OneHotCategorical, return_log_prob=True)
    return actor, logits_net


def make_critic(cfg):
    act = nn.ReLU if cfg["activation"] == "relu" else nn.Tanh
    return ValueOperator(mlp(1, cfg["hidden"], activation=act), in_keys=["observation"])


def _progress(algo, run_id, frames, total, log, extra=""):
    print(f"[{algo.upper()} run {run_id}] frames {frames:>7}/{total}  "
          f"mean reward (last 20 days) {log.last_mean():7.2f} {extra}")


# --------------------------------------------------------------------------- REINFORCE
def reward_to_go(rewards, dones, gamma):
    """Discounted return from each step to the end of its episode.
    Steps of an episode that is cut off at the end of the batch get mask=False."""
    T = rewards.shape[0]
    returns = torch.zeros(T)
    mask = torch.zeros(T, dtype=torch.bool)
    g, inside_complete = 0.0, False
    for t in reversed(range(T)):
        if dones[t]:
            g, inside_complete = 0.0, True
        g = rewards[t] + gamma * g
        returns[t] = g
        mask[t] = inside_complete
    return returns, mask


def train_reinforce(run_id, cfg, total_frames=TOTAL_FRAMES, seed=0, verbose=True):
    set_seed(seed)
    env = make_env(seed)
    actor, logits_net = make_actor(env, cfg)
    optim = torch.optim.Adam(actor.parameters(), lr=cfg["lr"])
    fpb = cfg["episodes_per_batch"] * DAY
    collector = Collector(lambda: make_env(seed), actor, frames_per_batch=fpb,
                          total_frames=total_frames, reset_at_each_iter=True)
    log = RunLogger("reinforce", run_id, cfg)
    frames = 0
    for i, data in enumerate(collector):
        frames += data.numel()
        log.episodes(finished_episode_rewards(data))
        rewards = data["next", "reward"].squeeze(-1)
        dones = data["next", "done"].squeeze(-1)
        returns, mask = reward_to_go(rewards, dones, cfg["gamma"])
        if mask.sum() == 0:
            continue
        adv = returns[mask]
        if cfg["normalize_returns"]:                 # baseline: centre and scale the returns
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        dist = actor.get_dist(data[mask])
        log_prob = dist.log_prob(data[mask]["action"])
        entropy = dist.entropy().mean()
        loss = -(log_prob * adv).mean() - cfg["entropy_coef"] * entropy
        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        optim.step()
        log.log("entropy", entropy.item())
        log.log("loss", loss.item())
        log.log("loss_frame", frames)
        if verbose and i % max(1, 40 // cfg["episodes_per_batch"]) == 0:
            _progress("reinforce", run_id, frames, total_frames, log, f"entropy {entropy.item():.3f}")
    collector.shutdown()
    return _finish("reinforce", run_id, cfg, logits_net, log, frames, verbose)


# --------------------------------------------------------------------------- PPO
def train_ppo(run_id, cfg, total_frames=TOTAL_FRAMES, seed=0, verbose=True):
    set_seed(seed)
    env = make_env(seed)
    actor, logits_net = make_actor(env, cfg)
    critic = make_critic(cfg)
    gae = GAE(gamma=cfg["gamma"], lmbda=cfg["gae_lambda"], value_network=critic, average_gae=True)
    loss_fn = ClipPPOLoss(actor, critic, clip_epsilon=cfg["clip_eps"], entropy_bonus=True,
                          entropy_coeff=cfg["entropy_coef"], critic_coeff=cfg["critic_coef"],
                          loss_critic_type="smooth_l1")
    optim = torch.optim.Adam(loss_fn.parameters(), lr=cfg["lr"])
    collector = Collector(lambda: make_env(seed), actor, frames_per_batch=cfg["frames_per_batch"],
                          total_frames=total_frames)
    log = RunLogger("ppo", run_id, cfg)
    frames = 0
    for i, data in enumerate(collector):
        frames += data.numel()
        log.episodes(finished_episode_rewards(data))
        for _ in range(cfg["epochs"]):
            with torch.no_grad():
                gae(data)
            flat = data.reshape(-1)
            perm = torch.randperm(flat.numel())
            for start in range(0, flat.numel(), cfg["minibatch"]):
                mb = flat[perm[start:start + cfg["minibatch"]]]
                out = loss_fn(mb)
                loss = out["loss_objective"] + out["loss_critic"] + out["loss_entropy"]
                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(loss_fn.parameters(), 0.5)
                optim.step()
        log.log("entropy", out["entropy"].item())
        log.log("loss", out["loss_objective"].item())
        log.log("critic_loss", out["loss_critic"].item())
        log.log("loss_frame", frames)
        if verbose and i % max(1, 14400 // cfg["frames_per_batch"]) == 0:
            _progress("ppo", run_id, frames, total_frames, log, f"entropy {out['entropy'].item():.3f}")
    collector.shutdown()
    return _finish("ppo", run_id, cfg, logits_net, log, frames, verbose)


# --------------------------------------------------------------------------- A2C
def train_a2c(run_id, cfg, total_frames=TOTAL_FRAMES, seed=0, verbose=True):
    set_seed(seed)
    env = make_env(seed)
    actor, logits_net = make_actor(env, cfg)
    critic = make_critic(cfg)
    gae = GAE(gamma=cfg["gamma"], lmbda=cfg["gae_lambda"], value_network=critic, average_gae=False)
    loss_fn = A2CLoss(actor, critic, entropy_bonus=True, entropy_coeff=cfg["entropy_coef"],
                      critic_coeff=cfg["critic_coef"], loss_critic_type="smooth_l1")
    optim = torch.optim.RMSprop(loss_fn.parameters(), lr=cfg["lr"], alpha=0.99, eps=1e-5)
    collector = Collector(lambda: make_env(seed), actor, frames_per_batch=cfg["n_steps"],
                          total_frames=total_frames)
    log = RunLogger("a2c", run_id, cfg)
    frames = 0
    for i, data in enumerate(collector):
        frames += data.numel()
        log.episodes(finished_episode_rewards(data))
        with torch.no_grad():
            gae(data)
        out = loss_fn(data)
        loss = out["loss_objective"] + out["loss_critic"] + out["loss_entropy"]
        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(loss_fn.parameters(), cfg["max_grad_norm"])
        optim.step()
        if i % max(1, 288 // cfg["n_steps"]) == 0:     # log about twice per simulated day
            log.log("entropy", out["entropy"].item())
            log.log("loss", out["loss_objective"].item())
            log.log("critic_loss", out["loss_critic"].item())
            log.log("loss_frame", frames)
        if verbose and i % max(1, 14400 // cfg["n_steps"]) == 0:
            _progress("a2c", run_id, frames, total_frames, log, f"entropy {out['entropy'].item():.3f}")
    collector.shutdown()
    return _finish("a2c", run_id, cfg, logits_net, log, frames, verbose)


def _finish(algo, run_id, cfg, logits_net, log, frames, verbose):
    evaluation = evaluate(logits_net)
    if verbose:
        print(f"[{algo.upper()} run {run_id}] evaluation: {evaluation}")
    result = log.save(evaluation, frames)
    save_model(logits_net, algo, "pg", cfg, evaluation, run_id)
    return result


TRAINERS = {"reinforce": train_reinforce, "ppo": train_ppo, "a2c": train_a2c}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["reinforce", "ppo", "a2c", "all"], default="all")
    p.add_argument("--run", type=int, default=None, help="run a single configuration (1-10)")
    p.add_argument("--frames", type=int, default=TOTAL_FRAMES)
    a = p.parse_args()
    algos = ["reinforce", "ppo", "a2c"] if a.algo == "all" else [a.algo]
    for algo in algos:
        base, sweep, columns = ALGOS[algo]
        ids = [a.run] if a.run else range(1, len(sweep) + 1)
        results = [TRAINERS[algo](i, {**base, **sweep[i - 1]}, a.frames) for i in ids]
        if a.run is None:
            write_sweep_table(algo, results, columns)


if __name__ == "__main__":
    main()
