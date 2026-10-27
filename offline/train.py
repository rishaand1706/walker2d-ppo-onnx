"""Train a policy from a fixed dataset, no environment interaction: behaviour cloning or TD3+BC.

    python -m offline.train --algo bc    --dataset medium-expert --seed 0
    python -m offline.train --algo td3bc --dataset medium-expert --seed 0 --steps 2000 --tag smoke

BC regresses the dataset's actions (MSE). TD3+BC (Fujimoto & Gu 2021) is TD3 (twin critics, target policy
smoothing, delayed actor updates) with a behaviour-cloning term in the actor loss,

    actor loss = -lambda * Q(s, pi(s)) + (pi(s) - a)^2,    lambda = alpha / mean|Q(s, pi(s))|,

so the policy may only improve on the data where the critic is confident. Both share the actor network, optimizer,
batch size and step budget, so the only difference is the critic. The final policy is scored on the evaluation
seeds; the periodic validation scores only draw the learning curve and never pick a checkpoint.

Writes runs/offline/<algo>_<dataset>_s<seed>/: actor.pt, curve.csv, result.json.
"""
import argparse
import copy
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from config import EVAL_EPISODES

from .common import (ALPHA, BATCH, GAMMA, LR, NOISE_CLIP, OUT, POLICY_DELAY, POLICY_NOISE, TAU, VAL_SEED, Actor,
                     Critic, StateNorm, evaluate, load_dataset, policy_fn)

STEPS = 300_000
VAL_EVERY, VAL_EPISODES = 25_000, 5


def soft_update(target, source):
    with torch.no_grad():
        for t, s in zip(target.parameters(), source.parameters()):
            t.mul_(1 - TAU).add_(TAU * s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=("bc", "td3bc"), required=True)
    ap.add_argument("--dataset", choices=("expert", "medium", "medium-expert"), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = OUT / (f"{args.algo}_{args.dataset}_s{args.seed}" + (f"_{args.tag}" if args.tag else ""))
    out.mkdir(parents=True, exist_ok=True)

    data = load_dataset(args.dataset)
    norm = StateNorm(data["obs"])
    obs = torch.as_tensor(norm(data["obs"]), dtype=torch.float32)
    next_obs = torch.as_tensor(norm(data["next_obs"]), dtype=torch.float32)
    act = torch.as_tensor(data["act"], dtype=torch.float32)
    rew = torch.as_tensor(data["rew"], dtype=torch.float32)[:, None]
    not_done = 1.0 - torch.as_tensor(data["terminal"], dtype=torch.float32)[:, None]
    n, obs_dim, act_dim = len(obs), obs.shape[1], act.shape[1]

    actor = Actor(obs_dim, act_dim)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=LR)
    if args.algo == "td3bc":
        actor_target = copy.deepcopy(actor)
        critic = Critic(obs_dim, act_dim)
        critic_target = copy.deepcopy(critic)
        critic_opt = torch.optim.Adam(critic.parameters(), lr=LR)

    curve, t0 = [], time.time()
    for step in range(1, args.steps + 1):
        idx = torch.as_tensor(rng.integers(0, n, BATCH))
        s, a = obs[idx], act[idx]

        if args.algo == "bc":
            loss = F.mse_loss(actor(s), a)
            actor_opt.zero_grad()
            loss.backward()
            actor_opt.step()
        else:
            with torch.no_grad():
                # Target policy smoothing: clipped noise on the target action, then the smaller of the two Qs.
                noise = (torch.randn_like(a) * POLICY_NOISE).clamp(-NOISE_CLIP, NOISE_CLIP)
                a_next = (actor_target(next_obs[idx]) + noise).clamp(-1.0, 1.0)
                q1_t, q2_t = critic_target(next_obs[idx], a_next)
                target = rew[idx] + not_done[idx] * GAMMA * torch.min(q1_t, q2_t)
            q1, q2 = critic(s, a)
            critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()

            if step % POLICY_DELAY == 0:
                pi = actor(s)
                q = critic(s, pi)[0]
                lam = ALPHA / q.abs().mean().detach()
                actor_loss = -lam * q.mean() + F.mse_loss(pi, a)
                actor_opt.zero_grad()
                actor_loss.backward()
                actor_opt.step()
                soft_update(critic_target, critic)
                soft_update(actor_target, actor)

        if step % VAL_EVERY == 0 or step == args.steps:
            eps = evaluate(policy_fn(actor, norm), VAL_EPISODES, base_seed=VAL_SEED)
            curve.append(dict(step=step, val_return=round(float(np.mean([e["ret"] for e in eps])), 1)))

    eps = evaluate(policy_fn(actor, norm), EVAL_EPISODES)
    result = dict(algo=args.algo, dataset=args.dataset, seed=args.seed, steps=args.steps,
                  eval_return=round(float(np.mean([e["ret"] for e in eps])), 1),
                  eval_std=round(float(np.std([e["ret"] for e in eps])), 1),
                  falls=sum(e["fell"] for e in eps), episodes=len(eps),
                  mean_distance=round(float(np.mean([e["distance"] for e in eps])), 2),
                  wall_minutes=round((time.time() - t0) / 60, 1))
    torch.save(actor.state_dict(), out / "actor.pt")
    with open(out / "curve.csv", "w") as f:
        f.write("step,val_return\n" + "".join(f"{c['step']},{c['val_return']}\n" for c in curve))
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"{out.name}: eval {result['eval_return']} +/- {result['eval_std']}, falls {result['falls']}/{len(eps)}, "
          f"{result['wall_minutes']} min")


if __name__ == "__main__":
    main()
