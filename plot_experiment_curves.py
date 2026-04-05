#!/usr/bin/env python3
"""
plot_experiment_curves.py  —  Per-experiment learning curves for thesis.

For each of the 4 experiments produces one figure showing:
  • Baseline (no feedback)
  • Human-only  (average of human individual curves)
  • LLM-only    (average of LLM individual curves)
  • Combined    (all annotators together)

Thesis-style: white background, clean spines, LaTeX math labels.

Usage
-----
  python plot_experiment_curves.py
  python plot_experiment_curves.py --window 30 --outdir results/plots/
"""

import argparse
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Data ─────────────────────────────────────────────────────────────────────

HUMAN_SESSIONS = {
    "entropy_fixed_on":   {"FAINT-ORBIT",  "GIANT-VESSEL",  "HOLLOW-BRIDGE"},
    "entropy_fixed_off":  {"FAINT-ORBIT",  "GIANT-VESSEL",  "HOLLOW-BRIDGE"},
    "entropy_random_on":  {"MISTY-CEDAR",  "NOBLE-PEBBLE"},
    "entropy_random_off": {"MISTY-CEDAR",  "NOBLE-PEBBLE"},
    "count_fixed_on":     {"AMBER-FOREST", "BRIGHT-CANYON", "CLEAN-VOYAGE"},
    "count_fixed_off":    {"AMBER-FOREST", "BRIGHT-CANYON", "CLEAN-VOYAGE"},
    "count_random_on":    {"IVORY-SPARK",  "JUMPY-CLOUD",   "LUCKY-BARREL"},
    "count_random_off":   {"IVORY-SPARK",  "JUMPY-CLOUD",   "LUCKY-BARREL"},
}

FILES = {
    "entropy_fixed_on":   "results/exp2_entropy_fixed.pkl",
    "entropy_fixed_off":  "results/exp2_cest_off.pkl",
    "entropy_random_on":  "results/exp4_entropy_random.pkl",
    "entropy_random_off": "results/exp4_cest_off.pkl",
    "count_fixed_on":     "results/exp1_cest_on.pkl",
    "count_fixed_off":    "results/exp1_cest_off.pkl",
    "count_random_on":    "results/exp3_count_random.pkl",
    "count_random_off":   "results/exp3_cest_off.pkl",
}

TITLES = {
    "entropy_fixed_on":   "Entropy Utility, Fixed Environment — $C_e$ ON",
    "entropy_fixed_off":  "Entropy Utility, Fixed Environment — $C_e$ OFF",
    "entropy_random_on":  "Entropy Utility, Randomised Environment — $C_e$ ON",
    "entropy_random_off": "Entropy Utility, Randomised Environment — $C_e$ OFF",
    "count_fixed_on":     "Count Utility, Fixed Environment — $C_e$ ON",
    "count_fixed_off":    "Count Utility, Fixed Environment — $C_e$ OFF",
    "count_random_on":    "Count Utility, Randomised Environment — $C_e$ ON",
    "count_random_off":   "Count Utility, Randomised Environment — $C_e$ OFF",
}

# Colour-blind safe, legible in greyscale print
C = {
    "baseline": ("#555555", (0, (5, 3))),   # grey, long-dash
    "human":    ("#d62728", "-"),            # red, solid
    "llm":      ("#2ca02c", "-"),            # green, solid
    "combined": ("#1f77b4", "-"),            # blue, solid
}
FILL_ALPHA = 0.10

# ── Helpers ──────────────────────────────────────────────────────────────────

def rolling_mean(x, w):
    x  = np.asarray(x, float)
    cs = np.cumsum(x)
    out = np.empty_like(x)
    out[:w] = cs[:w] / np.arange(1, w + 1)
    out[w:] = (cs[w:] - cs[:-w]) / w
    return out


def unpack(raw):
    arr = np.asarray(raw, float)
    if arr.ndim == 1:
        arr = arr[np.newaxis]
    return arr.mean(0), arr.std(0), arr


def group_curves(ind, sessions):
    """Average individual learning curves across a set of sessions."""
    keys = [s for s in ind if s in sessions]
    if not keys:
        return None, None
    n_trials = len(ind[keys[0]])
    n_ep     = len(ind[keys[0]][0])
    mat = np.zeros((n_trials, n_ep))
    for t in range(n_trials):
        mat[t] = np.mean([np.asarray(ind[k][t], float) for k in keys], axis=0)
    return mat.mean(0), mat.std(0)


def add_curve(ax, mean, std, w, color, ls, label, lw=1.8, zorder=4):
    sm = rolling_mean(mean, w)
    sd = rolling_mean(std,  w)
    ep = np.arange(1, len(sm) + 1)
    ax.plot(ep, sm, color=color, lw=lw, ls=ls, label=label, zorder=zorder)
    ax.fill_between(ep, sm - sd, sm + sd,
                    color=color, alpha=FILL_ALPHA, lw=0, zorder=zorder - 1)


def style_ax(ax):
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#aaaaaa")
    ax.spines["bottom"].set_color("#aaaaaa")
    ax.tick_params(axis="both", labelsize=9, color="#aaaaaa", length=3)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.tick_params(axis="y", which="minor", length=2, color="#dddddd")
    ax.grid(axis="y", color="#eeeeee", lw=0.7, ls="-", zorder=0)
    ax.set_axisbelow(True)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window",  type=int, default=30)
    ap.add_argument("--outdir",  default="results/plots/")
    args = ap.parse_args()
    W = args.window

    plt.rcParams.update({
        "font.family":    "sans-serif",
        "font.size":      10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
    })

    os.makedirs(args.outdir, exist_ok=True)

    for exp, pkl_path in FILES.items():
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)

        ind        = d.get("rewards_individual", {})
        human_sess = HUMAN_SESSIONS[exp]
        llm_sess   = set(d["sessions"]) - human_sess

        bl_mean, bl_std, _  = unpack(d["rewards_baseline"])
        fb_mean, fb_std, _  = unpack(d["rewards_feedback"])
        hm_mean, hm_std     = group_curves(ind, human_sess)
        lm_mean, lm_std     = group_curves(ind, llm_sess)

        fig, ax = plt.subplots(figsize=(7, 4.2), facecolor="white")
        style_ax(ax)

        add_curve(ax, bl_mean, bl_std, W,
                  *C["baseline"], "Baseline (no feedback)", lw=1.6, zorder=3)
        if hm_mean is not None:
            add_curve(ax, hm_mean, hm_std, W,
                      *C["human"], "Human annotators", lw=1.6, zorder=4)
        if lm_mean is not None:
            add_curve(ax, lm_mean, lm_std, W,
                      *C["llm"], "LLM annotators", lw=1.6, zorder=4)
        add_curve(ax, fb_mean, fb_std, W,
                  *C["combined"], "Combined (human + LLM)", lw=2.2, zorder=6)

        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.set_title(TITLES[exp], pad=8)
        ax.legend(
            frameon=True, framealpha=0.95,
            edgecolor="#cccccc", loc="lower right",
        )

        fig.tight_layout()
        out = os.path.join(args.outdir, f"{exp}_learning_curves.png")
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Saved → {out}")
        plt.close(fig)


if __name__ == "__main__":
    main()
