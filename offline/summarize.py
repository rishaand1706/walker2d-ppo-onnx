"""Tables and the figure for the offline results.

    python -m offline.summarize     # after offline/run_all.sh

Writes results/offline_runs.csv (one row per run), results/offline_summary.json (per dataset and algorithm, over
seeds) and results/offline_curves.png with its _dark twin. The normalized score is D4RL's convention on our own
reference points: 100 * (return - random) / (expert - random), with the random policy and the deterministic expert
scored on the same evaluation seeds as every offline policy (results/offline_data.json).
"""
import csv
import json

import matplotlib.pyplot as plt
import numpy as np

from config import RESULTS
from plot import THEMES, style

from .common import DATASETS, OUT

ALGOS = ("bc", "td3bc")
NAME = dict(bc="BC", td3bc="TD3+BC")
COLOR = dict(light=dict(bc="#6250d6", td3bc="#e87ba4"), dark=dict(bc="#9085e9", td3bc="#d55181"))


def load():
    runs = []
    for d in sorted(OUT.iterdir()):
        f = d / "result.json"
        if f.exists() and "_smoke" not in d.name:
            r = json.loads(f.read_text())
            r["curve"] = [tuple(map(float, line.split(","))) for line in (d / "curve.csv").read_text().split()[1:]]
            runs.append(r)
    return runs


def main():
    runs = load()
    ref = json.loads((RESULTS / "offline_data.json").read_text())
    lo, hi = ref["reference"]["random_return"], ref["reference"]["expert_return"]
    score = lambda r: 100 * (r - lo) / (hi - lo)

    def data_average(dataset):
        srcs = DATASETS[dataset]
        eps = sum(ref[s]["episodes"] for s in srcs)
        return sum(ref[s]["mean_episode_return"] * ref[s]["episodes"] for s in srcs) / eps

    with open(RESULTS / "offline_runs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algo", "dataset", "seed", "eval_return", "normalized", "falls", "episodes", "mean_distance"])
        for r in runs:
            w.writerow([r["algo"], r["dataset"], r["seed"], r["eval_return"], round(score(r["eval_return"]), 1),
                        r["falls"], r["episodes"], r["mean_distance"]])

    summary = {}
    for ds in DATASETS:
        summary[ds] = dict(data_average_return=round(data_average(ds), 1),
                           data_average_normalized=round(score(data_average(ds)), 1))
        for a in ALGOS:
            rs = [r for r in runs if r["dataset"] == ds and r["algo"] == a]
            if not rs:
                continue
            ret = np.array([r["eval_return"] for r in rs])
            summary[ds][a] = dict(seeds=len(rs), mean_return=round(float(ret.mean()), 1),
                                  std_across_seeds=round(float(ret.std(ddof=1)), 1) if len(rs) > 1 else None,
                                  normalized=round(score(float(ret.mean())), 1),
                                  per_seed=[r["eval_return"] for r in sorted(rs, key=lambda r: r["seed"])],
                                  falls=f"{sum(r['falls'] for r in rs)}/{sum(r['episodes'] for r in rs)}")
    summary["reference"] = ref["reference"]
    (RESULTS / "offline_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    for mode, t in THEMES.items():
        fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.9), dpi=160, sharey=True, facecolor=t["surface"])
        for ax, ds in zip(axes, DATASETS):
            style(ax, t)
            for y, label in ((hi, "expert"), (data_average(ds), "data average")):
                ax.axhline(y, color=t["muted"], lw=1, ls=(0, (4, 3)))
                ax.text(0.01, y, label, transform=ax.get_yaxis_transform(), va="bottom", color=t["muted"], fontsize=8)
            for a in ALGOS:
                curves = [r["curve"] for r in runs if r["dataset"] == ds and r["algo"] == a]
                if not curves:
                    continue
                x = np.array([c[0] for c in curves[0]]) / 1e3
                ys = np.array([[c[1] for c in cur] for cur in curves])
                for y in ys:
                    ax.plot(x, y, color=COLOR[mode][a], lw=1, alpha=0.3)
                ax.plot(x, ys.mean(0), color=COLOR[mode][a], lw=2.2, label=NAME[a])
            ax.set_title(ds, color=t["ink"], fontsize=10, loc="left")
            ax.set_xlabel("gradient steps (thousands)", color=t["ink2"])
        axes[0].set_ylabel("validation return", color=t["ink2"])
        handles, labels = axes[0].get_legend_handles_labels()
        leg = fig.legend(handles, [f"{l} (mean; thin = each seed)" for l in labels], loc="upper left",
                         bbox_to_anchor=(0.01, 0.92), ncol=2, frameon=False, fontsize=9)
        for text in leg.get_texts():
            text.set_color(t["ink"])
        fig.suptitle("Offline learning from PPO rollouts, 5 seeds per algorithm and dataset", x=0.01, ha="left",
                     color=t["ink"], fontsize=11)
        fig.subplots_adjust(left=0.07, right=0.99, top=0.76, bottom=0.15, wspace=0.08)
        fig.savefig(RESULTS / f"offline_curves{'' if mode == 'light' else '_dark'}.png", facecolor=t["surface"])
        plt.close(fig)

    for ds in DATASETS:
        s = summary[ds]
        line = f"{ds:13s} data {s['data_average_return']:7.1f}"
        for a in ALGOS:
            if a in s:
                line += f" | {NAME[a]} {s[a]['mean_return']:7.1f} +/- {s[a]['std_across_seeds']} ({s[a]['normalized']}) falls {s[a]['falls']}"
        print(line)


if __name__ == "__main__":
    main()
