#!/usr/bin/env python3
"""
plot_cest_curves.py — Learning curves: Ce ON vs Ce OFF across all 4 experiments.

2×2 grid of subplots, one per experiment.  Academic / thesis style.

Each panel shows:
  • Baseline (no feedback)        — grey
  • Combined  Ce ON               — solid blue
  • Combined  Ce OFF              — solid red/orange
  • Human-only Ce ON / OFF        — thin blue solid/dashed  (optional)
  • LLM-only   Ce ON / OFF        — thin green solid/dashed (optional)

Usage
-----
  python plot_cest_curves.py
  python plot_cest_curves.py --window 30 --save results/plots/cest_learning_curves.png
  python plot_cest_curves.py --subgroups --save results/plots/cest_learning_curves_full.png
"""

import argparse
import pickle
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Session groupings ────────────────────────────────────────────────────────

HUMAN = {
    "exp1": {"AMBER-FOREST", "BRIGHT-CANYON", "CLEAN-VOYAGE"},
    "exp2": {"FAINT-ORBIT", "GIANT-VESSEL", "HOLLOW-BRIDGE"},
    "exp3": {"IVORY-SPARK", "JUMPY-CLOUD", "LUCKY-BARREL"},
    "exp4": {"MISTY-CEDAR", "NOBLE-PEBBLE"},
}

FILES = {
    "exp1": ("results/exp1_cest_on.pkl",       "results/exp1_cest_off.pkl"),
    "exp2": ("results/exp2_entropy_fixed.pkl",  "results/exp2_cest_off.pkl"),
    "exp3": ("results/exp3_count_random.pkl",   "results/exp3_cest_off.pkl"),
    "exp4": ("results/exp4_entropy_random.pkl", "results/exp4_cest_off.pkl"),
}

TITLES = {
    "exp1": "(a)  Exp 1 — Count / Fixed",
    "exp2": "(b)  Exp 2 — Entropy / Fixed",
    "exp3": "(c)  Exp 3 — Count / Random",
    "exp4": "(d)  Exp 4 — Entropy / Random",
}

# ── Thesis-friendly colour palette (works on white) ─────────────────────────
# Colour-blind safe: distinguishable in greyscale print too

C_BASELINE = "#555555"          # mid-grey
C_ON_COMB  = "#1f77b4"          # mpl blue   — Ce ON combined
C_OFF_COMB = "#d62728"          # mpl red    — Ce OFF combined
C_ON_HUM   = "#2ca02c"          # mpl green  — Ce ON human-only
C_OFF_HUM  = "#2ca02c"          # same hue, dashed
C_ON_LLM   = "#9467bd"          # mpl purple — Ce ON LLM-only
C_OFF_LLM  = "#9467bd"          # same hue, dashed

FILL_ALPHA  = 0.12
FILL_ALPHA2 = 0.07              # sub-group fills

# ── Helpers ──────────────────────────────────────────────────────────────────

def rolling_mean(x, w):
    x  = np.array(x, dtype=float)
    cs = np.cumsum(x)
    out = np.empty_like(x)
    out[:w] = cs[:w] / np.arange(1, w + 1)
    out[w:] = (cs[w:] - cs[:-w]) / w
    return out


def unpack(raw):
    arr = np.array(raw, dtype=float)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    return arr.mean(axis=0), arr.std(axis=0), arr


def group_curves(rewards_individual, sessions):
    sess_list = [s for s in rewards_individual if s in sessions]
    if not sess_list:
        return None, None, None
    n_trials = len(rewards_individual[sess_list[0]])
    n_ep     = len(rewards_individual[sess_list[0]][0])
    trials   = np.zeros((n_trials, n_ep))
    for t in range(n_trials):
        for s in sess_list:
            trials[t] += np.array(rewards_individual[s][t], dtype=float)
        trials[t] /= len(sess_list)
    return trials.mean(axis=0), trials.std(axis=0), trials


def plot_curve(ax, mean, std, w, color, label,
               lw=1.8, ls="-", alpha_fill=FILL_ALPHA, zorder=4):
    sm  = rolling_mean(mean, w)
    sd  = rolling_mean(std,  w)
    eps = np.arange(1, len(sm) + 1)
    ax.plot(eps, sm, color=color, linewidth=lw, linestyle=ls,
            label=label, zorder=zorder)
    ax.fill_between(eps, sm - sd, sm + sd,
                    color=color, alpha=alpha_fill, linewidth=0, zorder=zorder - 1)


def style_ax(ax, ylabel=False):
    """Apply clean academic axes style."""
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")
    ax.tick_params(axis="both", which="major", labelsize=8,
                   color="#888888", length=3)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.tick_params(axis="y", which="minor", length=2, color="#cccccc")
    ax.grid(axis="y", color="#e0e0e0", linewidth=0.6, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("Episode", fontsize=9)
    if ylabel:
        ax.set_ylabel("Total Reward", fontsize=9)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=30,
                    help="Smoothing window in episodes (default: 30)")
    ap.add_argument("--save", default="results/plots/cest_learning_curves.png")
    ap.add_argument("--subgroups", action="store_true",
                    help="Also plot human-only and LLM-only curves")
    args = ap.parse_args()
    W = args.window

    # Use a clean sans-serif font throughout
    plt.rcParams.update({
        "font.family":      "sans-serif",
        "font.size":        9,
        "axes.titlesize":   10,
        "axes.labelsize":   9,
        "legend.fontsize":  8.5,
        "figure.dpi":       150,
    })

    fig, axes = plt.subplots(
        2, 2, figsize=(11, 7),
        facecolor="white",
        sharex=False, sharey=False,
    )
    fig.subplots_adjust(hspace=0.42, wspace=0.30,
                        left=0.08, right=0.98, top=0.88, bottom=0.14)

    for idx, (ax, (exp, (on_f, off_f))) in enumerate(
            zip(axes.flat, FILES.items())):

        style_ax(ax, ylabel=(idx % 2 == 0))

        with open(on_f,  "rb") as f: don  = pickle.load(f)
        with open(off_f, "rb") as f: doff = pickle.load(f)

        human_sess = HUMAN[exp]
        llm_sess   = set(don["sessions"]) - human_sess

        # Baseline — pool both runs' baseline seeds for lower variance
        _, _, bl_t_on  = unpack(don["rewards_baseline"])
        _, _, bl_t_off = unpack(doff["rewards_baseline"])
        bl_trials = np.vstack([bl_t_on, bl_t_off])
        plot_curve(ax, bl_trials.mean(0), bl_trials.std(0), W,
                   C_BASELINE, "Baseline (no feedback)",
                   lw=1.5, ls=(0, (4, 2)), zorder=3)

        # Combined Ce ON / OFF
        fb_m_on,  fb_s_on,  _ = unpack(don["rewards_feedback"])
        fb_m_off, fb_s_off, _ = unpack(doff["rewards_feedback"])
        plot_curve(ax, fb_m_on,  fb_s_on,  W, C_ON_COMB,
                   "Combined  $C_e$ ON",  lw=2.2, zorder=6)
        plot_curve(ax, fb_m_off, fb_s_off, W, C_OFF_COMB,
                   "Combined  $C_e$ OFF", lw=2.2, ls="--", zorder=5)

        if args.subgroups:
            ind_on  = don.get("rewards_individual",  {})
            ind_off = doff.get("rewards_individual", {})

            for (sess_set, c_on, c_off, lbl) in [
                (human_sess, C_ON_HUM, C_OFF_HUM, "Human"),
                (llm_sess,   C_ON_LLM, C_OFF_LLM, "LLM"),
            ]:
                hm_on,  hs_on,  _ = group_curves(ind_on,  sess_set)
                hm_off, hs_off, _ = group_curves(ind_off, sess_set)
                if hm_on is not None:
                    plot_curve(ax, hm_on,  hs_on,  W, c_on,
                               f"{lbl}  $C_e$ ON",
                               lw=1.1, alpha_fill=FILL_ALPHA2, zorder=4)
                if hm_off is not None:
                    plot_curve(ax, hm_off, hs_off, W, c_off,
                               f"{lbl}  $C_e$ OFF",
                               lw=1.1, ls="--", alpha_fill=FILL_ALPHA2, zorder=3)

        ax.set_title(TITLES[exp], fontsize=10, loc="left", pad=5)

    # Single shared legend centred below all panels
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=3,
        frameon=True,
        framealpha=0.95,
        edgecolor="#cccccc",
        fontsize=8.5,
        bbox_to_anchor=(0.5, 0.0),
    )

    fig.suptitle(
        "Effect of Adaptive Consistency Estimation on Learning: "
        "$C_e$ ON vs. $C_e$ OFF",
        fontsize=11, y=0.96,
    )

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    fig.savefig(args.save, dpi=200, bbox_inches="tight",
                facecolor="white")
    print(f"Saved → {args.save}")
    plt.close(fig)


if __name__ == "__main__":
    main()
