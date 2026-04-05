#!/usr/bin/env python3
"""
plot_scheme_curves.py — Learning curve comparison of alternative training schemes.

Produces one figure per experiment (4 figures total).  Each figure shows:
  • Baseline (no feedback)           — grey dashed
  • Standard Ce ON combined          — blue solid   (from exp{N}_cest_on.pkl)
  • Burn-in                          — teal
  • Two-phase (LLM boot → combined)  — orange
  • Warm Q-init                      — purple
  • Disagreement replay              — green
  • Majority vote                    — red

Academic thesis style: white background, clean spines.

Usage
-----
  python plot_scheme_curves.py
  python plot_scheme_curves.py --window 30 --outdir results/plots/
"""

import argparse
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ── File paths ────────────────────────────────────────────────────────────────

EXPERIMENTS = {
    "exp1": {
        "label":    "Exp 1 — Count / Fixed",
        "baseline": "results/exp1_cest_on.pkl",        # used for its baseline trials
        "cest_on":  "results/exp1_cest_on.pkl",
        "schemes": {
            "burnin":        "results/schemes/exp1_burnin.pkl",
            "two-phase":     "results/schemes/exp1_two_phase.pkl",
            "warm-init":     "results/schemes/exp1_warm_init.pkl",
            "disagreement":  "results/schemes/exp1_disagreement.pkl",
            "majority-vote": "results/schemes/exp1_majority_vote.pkl",
        },
    },
    "exp2": {
        "label":    "Exp 2 — Entropy / Fixed",
        "baseline": "results/exp2_entropy_fixed.pkl",
        "cest_on":  "results/exp2_entropy_fixed.pkl",
        "schemes": {
            "burnin":        "results/schemes/exp2_burnin.pkl",
            "two-phase":     "results/schemes/exp2_two_phase.pkl",
            "warm-init":     "results/schemes/exp2_warm_init.pkl",
            "disagreement":  "results/schemes/exp2_disagreement.pkl",
            "majority-vote": "results/schemes/exp2_majority_vote.pkl",
        },
    },
    "exp3": {
        "label":    "Exp 3 — Count / Random",
        "baseline": "results/exp3_count_random.pkl",
        "cest_on":  "results/exp3_count_random.pkl",
        "schemes": {
            "burnin":        "results/schemes/exp3_burnin.pkl",
            "two-phase":     "results/schemes/exp3_two_phase.pkl",
            "warm-init":     "results/schemes/exp3_warm_init.pkl",
            "disagreement":  "results/schemes/exp3_disagreement.pkl",
            "majority-vote": "results/schemes/exp3_majority_vote.pkl",
        },
    },
    "exp4": {
        "label":    "Exp 4 — Entropy / Random",
        "baseline": "results/exp4_entropy_random.pkl",
        "cest_on":  "results/exp4_entropy_random.pkl",
        "schemes": {
            "burnin":        "results/schemes/exp4_burnin.pkl",
            "two-phase":     "results/schemes/exp4_two_phase.pkl",
            "warm-init":     "results/schemes/exp4_warm_init.pkl",
            "disagreement":  "results/schemes/exp4_disagreement.pkl",
            "majority-vote": "results/schemes/exp4_majority_vote.pkl",
        },
    },
}

# Colour-blind safe palette
COLOURS = {
    "baseline":      ("#555555", (0, (5, 3)), 1.5),  # grey, long-dash
    "cest-on":       ("#1f77b4", "-",         2.2),  # blue, bold
    "burnin":        ("#17becf", "-",         1.8),  # teal
    "two-phase":     ("#ff7f0e", "-",         1.8),  # orange
    "warm-init":     ("#9467bd", "-",         1.8),  # purple
    "disagreement":  ("#2ca02c", "-",         1.8),  # green
    "majority-vote": ("#d62728", "-",         1.8),  # red
}

LABELS = {
    "baseline":      "Baseline (no feedback)",
    "cest-on":       "Standard  $C_e$ ON",
    "burnin":        "Burn-in",
    "two-phase":     "Two-phase (LLM → combined)",
    "warm-init":     "Warm Q-init",
    "disagreement":  "Disagreement replay",
    "majority-vote": "Majority vote",
}

FILL_ALPHA = 0.09


# ── Helpers ───────────────────────────────────────────────────────────────────

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
    return arr.mean(0), arr.std(0)


def add_curve(ax, mean, std, w, color, ls, lw, label, zorder=4):
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window",  type=int, default=30,
                    help="Smoothing window in episodes (default 30)")
    ap.add_argument("--outdir",  default="results/plots/",
                    help="Output directory for figures")
    ap.add_argument("--combined", action="store_true",
                    help="Also produce a single 2×2 combined figure")
    args = ap.parse_args()
    W = args.window

    plt.rcParams.update({
        "font.family":    "sans-serif",
        "font.size":      10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
    })

    os.makedirs(args.outdir, exist_ok=True)

    for exp_id, cfg in EXPERIMENTS.items():
        # Check which scheme files exist
        available = {}
        for scheme, path in cfg["schemes"].items():
            if os.path.exists(path):
                available[scheme] = path
            else:
                print(f"  [skip] {path} not found")

        if not available:
            print(f"No scheme results found for {exp_id} — skipping")
            continue

        fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="white")
        style_ax(ax)

        # Baseline — from the Ce ON file (pooled trials)
        with open(cfg["cest_on"], "rb") as f:
            d_ref = pickle.load(f)
        bl_m, bl_s = unpack(d_ref["rewards_baseline"])
        add_curve(ax, bl_m, bl_s, W, *COLOURS["baseline"],
                  LABELS["baseline"], zorder=3)

        # Standard Ce ON combined
        fb_m, fb_s = unpack(d_ref["rewards_feedback"])
        add_curve(ax, fb_m, fb_s, W, *COLOURS["cest-on"],
                  LABELS["cest-on"], zorder=6)

        # Each scheme
        z = 5
        for scheme in ["burnin", "two-phase", "warm-init",
                       "disagreement", "majority-vote"]:
            if scheme not in available:
                continue
            with open(available[scheme], "rb") as f:
                d = pickle.load(f)
            m, s = unpack(d["rewards_feedback"])
            add_curve(ax, m, s, W, *COLOURS[scheme], LABELS[scheme], zorder=z)
            z -= 1

        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.set_title(cfg["label"] + " — Training Scheme Comparison", pad=8)
        ax.legend(
            frameon=True, framealpha=0.95,
            edgecolor="#cccccc", loc="lower right",
            fontsize=8,
        )
        fig.tight_layout()
        out = os.path.join(args.outdir, f"{exp_id}_scheme_comparison.png")
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Saved → {out}")
        plt.close(fig)

    # ── Optional 2×2 overview ────────────────────────────────────────────────
    if args.combined:
        fig2, axes = plt.subplots(2, 2, figsize=(13, 8), facecolor="white")
        fig2.subplots_adjust(hspace=0.40, wspace=0.28,
                             left=0.08, right=0.98, top=0.88, bottom=0.14)

        for idx, (ax, (exp_id, cfg)) in enumerate(
                zip(axes.flat, EXPERIMENTS.items())):
            style_ax(ax)
            ax.set_xlabel("Episode", fontsize=9)
            if idx % 2 == 0:
                ax.set_ylabel("Total Reward", fontsize=9)

            if not os.path.exists(cfg["cest_on"]):
                continue

            with open(cfg["cest_on"], "rb") as f:
                d_ref = pickle.load(f)
            bl_m, bl_s = unpack(d_ref["rewards_baseline"])
            add_curve(ax, bl_m, bl_s, W, *COLOURS["baseline"],
                      LABELS["baseline"], zorder=3)
            fb_m, fb_s = unpack(d_ref["rewards_feedback"])
            add_curve(ax, fb_m, fb_s, W, *COLOURS["cest-on"],
                      LABELS["cest-on"], zorder=6)

            z = 5
            for scheme in ["burnin", "two-phase", "warm-init",
                           "disagreement", "majority-vote"]:
                path = cfg["schemes"].get(scheme)
                if path and os.path.exists(path):
                    with open(path, "rb") as f:
                        d = pickle.load(f)
                    m, s = unpack(d["rewards_feedback"])
                    add_curve(ax, m, s, W, *COLOURS[scheme], LABELS[scheme],
                              zorder=z)
                    z -= 1

            ax.set_title(cfg["label"], fontsize=10, loc="left", pad=5)

        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig2.legend(
            handles, labels,
            loc="lower center", ncol=4,
            frameon=True, framealpha=0.95,
            edgecolor="#cccccc", fontsize=8.5,
            bbox_to_anchor=(0.5, 0.0),
        )
        fig2.suptitle(
            "Alternative Training Scheme Comparison across All Experiments",
            fontsize=11, y=0.96,
        )
        out2 = os.path.join(args.outdir, "all_schemes_overview.png")
        fig2.savefig(out2, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Saved → {out2}")
        plt.close(fig2)


if __name__ == "__main__":
    main()
