"""README figures: training curves for all three conditions, and the robustness sweep (norm vs dr).

    python plot.py      # after train (runs/) and evaluate.py (results/robustness.csv)

Writes results/learning_curves.png and results/robustness.png, each with a _dark twin rendered from the dark
palette (a <picture> tag in the README picks one), never an inverted copy. Colors follow the condition, not its
rank: norm is always blue, raw orange, dr aqua.
"""
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from config import CONDITIONS, DR_FRICTION, DR_MASS, RESULTS, RUNS, SEEDS, TIMESTEPS  # noqa: E402

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", band="#f0efec", series=dict(norm="#2a78d6", raw="#eb6834", dr="#1baf7a")),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", band="#262624", series=dict(norm="#3987e5", raw="#d95926", dr="#199e70")),
}
LABEL = dict(norm="normalized", raw="no normalization", dr="normalized + domain randomization")
SHORT = dict(norm="normalized", raw="no normalization", dr="normalized + DR")
BIN = 25_000
plt.rcParams.update({"font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 10})


def style(ax, t):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["axis"])
    ax.grid(axis="y", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=t["muted"], labelcolor=t["ink2"], length=3)


def binned_returns(run_dir):
    """Mean training-episode return in each BIN of environment steps (NaN where no episode ended)."""
    rows = list(csv.reader(open(run_dir / "train.monitor.csv")))[2:]
    ret = np.array([float(r[0]) for r in rows])
    end = np.cumsum([int(r[1]) for r in rows])
    which = np.minimum(end // BIN, TIMESTEPS // BIN - 1)
    return np.array([ret[which == b].mean() if (which == b).any() else np.nan for b in range(TIMESTEPS // BIN)])


def spread_labels(ys, gap):
    """Nudge end-of-line label heights apart so none overlap; keeps their order."""
    order = np.argsort(ys)
    out = np.array(ys, dtype=float)
    for a, b in zip(order, order[1:]):
        out[b] = max(out[b], out[a] + gap)
    return out


def learning_curves(mode):
    t = THEMES[mode]
    fig, ax = plt.subplots(figsize=(8.6, 4.4), dpi=160, facecolor=t["surface"])
    style(ax, t)
    x = (np.arange(TIMESTEPS // BIN) + 1) * BIN / 1e6
    ends = {}
    for c in CONDITIONS:
        curves = np.array([binned_returns(RUNS / f"{c}_s{s}") for s in SEEDS])
        mean, sd = np.nanmean(curves, 0), np.nanstd(curves, 0)
        ax.fill_between(x, np.maximum(mean - sd, 0), mean + sd, color=t["series"][c],
                        alpha=0.12 if mode == "light" else 0.18, lw=0)
        ax.plot(x, mean, color=t["series"][c], lw=2, label=LABEL[c])
        ends[c] = mean[-1]
    ylo, yhi = ax.get_ylim()
    for c, y in zip(ends, spread_labels(list(ends.values()), gap=0.055 * (yhi - ylo))):
        ax.text(x[-1] + 0.015, y, f"{SHORT[c]}  {ends[c]:,.0f}", color=t["ink"], va="center", fontsize=9)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("environment steps (millions)", color=t["ink2"])
    ax.set_ylabel("training episode return", color=t["ink2"])
    ax.set_title("PPO on Walker2d-v5: training return, mean of 5 seeds (band = ±1 s.d.)",
                 loc="left", color=t["ink"], fontsize=11, pad=12)
    leg = ax.legend(loc="upper left", frameon=False, fontsize=9)
    for text in leg.get_texts():
        text.set_color(t["ink"])
    fig.text(0.01, 0.01, "Domain-randomized runs train under random mass and friction, so their training return "
             "is not measured on the nominal robot.", color=t["muted"], fontsize=8)
    fig.subplots_adjust(left=0.08, right=0.77, top=0.88, bottom=0.17)
    fig.savefig(RESULTS / f"learning_curves{'' if mode == 'light' else '_dark'}.png", facecolor=t["surface"])
    plt.close(fig)


def robustness(mode):
    t = THEMES[mode]
    rows = list(csv.DictReader(open(RESULTS / "robustness.csv")))
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9), dpi=160, sharey=True, facecolor=t["surface"])
    for ax, param, trained in zip(axes, ("mass", "friction"), (DR_MASS, DR_FRICTION)):
        style(ax, t)
        ax.axvspan(*trained, color=t["band"], lw=0, zorder=0)
        for c in ("norm", "dr"):
            pts = sorted({float(r["scale"]) for r in rows if r["param"] == param})
            per = np.array([[float(r["mean_return"]) for r in rows
                             if r["param"] == param and r["condition"] == c and float(r["scale"]) == k] for k in pts])
            for seed_line in per.T:  # each seed, thin: the spread is bimodal, so a +/-1 s.d. band would mislead
                ax.plot(pts, seed_line, color=t["series"][c], lw=1, alpha=0.35)
            ax.plot(pts, per.mean(1), color=t["series"][c], lw=2.2, marker="o", markersize=5,
                    markeredgecolor=t["surface"], markeredgewidth=1.5, label=f"{LABEL[c]} (mean; thin = each seed)")
        ax.set_xlabel(f"{param} scale (1.0 = nominal robot)", color=t["ink2"])
        ax.text(sum(trained) / 2, 1.0, "randomized in training", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", color=t["muted"], fontsize=8)
    axes[0].set_ylabel("eval return (deterministic)", color=t["ink2"])
    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.93), ncol=2, frameon=False, fontsize=9)
    for text in leg.get_texts():
        text.set_color(t["ink"])
    fig.suptitle("Robustness: body mass and ground friction scaled one at a time, 5 seeds per condition",
                 x=0.01, ha="left", color=t["ink"], fontsize=11)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.78, bottom=0.15, wspace=0.08)
    fig.savefig(RESULTS / f"robustness{'' if mode == 'light' else '_dark'}.png", facecolor=t["surface"])
    plt.close(fig)


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    for mode in THEMES:
        learning_curves(mode)
        robustness(mode)
    print("wrote results/learning_curves*.png and results/robustness*.png")
