#!/usr/bin/env python3
"""
view_all_experiments.py — Visualise all four offline_train.py result files together.

Layout
------
  Row 1 (2 panels):  Learning curves  — Exp 1 & 2 (fixed env)
  Row 2 (2 panels):  Learning curves  — Exp 3 & 4 (random env)
  Row 3 (1 wide):    Final-reward bar chart — all 4 experiments, per-participant
  Row 4 (1 wide):    Ce scores — all 4 experiments, per-participant

Usage
-----
  python view_all_experiments.py \\
      results/exp1_count_fixed.pkl \\
      results/exp2_entropy_fixed.pkl \\
      results/exp3_count_random.pkl \\
      results/exp4_entropy_random.pkl

  # Custom labels:
  python view_all_experiments.py exp1.pkl exp2.pkl exp3.pkl exp4.pkl \\
      --labels "Count/Fixed" "Entropy/Fixed" "Count/Random" "Entropy/Random"

  # Save to file:
  python view_all_experiments.py exp1.pkl exp2.pkl exp3.pkl exp4.pkl \\
      --save results/plots/all_experiments.png
"""

import argparse
import os
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_LABELS = [
    "Exp 1 · Count / Fixed",
    "Exp 2 · Entropy / Fixed",
    "Exp 3 · Count / Random",
    "Exp 4 · Entropy / Random",
]

PALETTE = {
    "baseline":  "#555555",
    "combined":  "#2196F3",   # blue
    "exp":       ["#E53935", "#43A047", "#FB8C00", "#8E24AA"],  # red, green, orange, purple
    "human":     "#26C6DA",
    "llm":       "#FFA726",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load(path: str) -> dict:
    if not os.path.exists(path):
        print(f"  Warning: {path} not found — skipping.")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def smooth(arr, w: int = 20) -> np.ndarray:
    """Simple moving average."""
    if len(arr) < w:
        return np.array(arr, dtype=float)
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode="valid")


def mean_trials(list_of_lists) -> np.ndarray:
    """Mean across trials (list of equal-length arrays)."""
    return np.mean(list_of_lists, axis=0)


def tail_mean(arr, w: int = 50) -> float:
    return float(np.mean(arr[-w:])) if len(arr) >= w else float(np.mean(arr))


def is_llm(session: str) -> bool:
    """Heuristic: LLM sessions use all-caps hyphen codes like BRAVE-COMPASS; they also
    contain no lower-case so we distinguish them from old-style LLM_<name> keys."""
    return session.startswith("LLM_") or (
        "-" in session and session.replace("-", "").isupper()
    )


# ── Plot helpers ──────────────────────────────────────────────────────────────

def plot_learning_curve(ax, results: dict, label: str, color: str, window: int = 20):
    """Draw smoothed combined + baseline curves on ax."""
    if results is None:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        ax.set_title(label)
        return

    fb_mean  = mean_trials(results["rewards_feedback"])
    bl_mean  = mean_trials(results["rewards_baseline"])
    fb_s     = smooth(fb_mean, window)
    bl_s     = smooth(bl_mean, window)

    # shade std across trials if >1 trial
    if len(results["rewards_feedback"]) > 1:
        fb_std = np.std(results["rewards_feedback"], axis=0)
        bl_std = np.std(results["rewards_baseline"], axis=0)
        fb_std_s = smooth(fb_std, window)
        bl_std_s = smooth(bl_std, window)
        x_fb = np.arange(len(fb_s))
        x_bl = np.arange(len(bl_s))
        ax.fill_between(x_fb, fb_s - fb_std_s, fb_s + fb_std_s, alpha=0.15, color=color)
        ax.fill_between(x_bl, bl_s - bl_std_s, bl_s + bl_std_s, alpha=0.10, color=PALETTE["baseline"])

    ax.plot(bl_s, color=PALETTE["baseline"], linewidth=1.2, linestyle="--", label="Baseline (no FB)")
    ax.plot(fb_s, color=color, linewidth=2.0, label="Combined feedback")

    # Individual participants
    for sh in results.get("sessions", []):
        ind = results.get("rewards_individual", {}).get(sh)
        if ind:
            ind_mean = mean_trials(ind)
            ind_s    = smooth(ind_mean, window)
            c = PALETTE["llm"] if is_llm(sh) else PALETTE["human"]
            ax.plot(ind_s, color=c, linewidth=0.7, alpha=0.5)

    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend(fontsize=7, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)


def plot_final_bar(ax, all_results: list, labels: list, window: int = 50):
    """Grouped bar chart comparing baseline vs combined across all experiments.

    Participants are NOT compared across experiments — each person/LLM only ran
    one experiment, so cross-experiment individual comparisons are meaningless.
    Instead this shows the aggregate effect of feedback (combined) vs no feedback
    (baseline) for each experiment, plus per-participant bars within each group.

    Within each experiment group the bars are:
      baseline | combined | human_1 … human_N | llm_1 … llm_M
    All bars belong to that experiment only — no participant appears in a group
    they didn't run.
    """
    from matplotlib.patches import Patch

    valid = [(r, lbl) for r, lbl in zip(all_results, labels) if r is not None]
    if not valid:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    n_groups   = len(valid)
    # Max bars per group = baseline + combined + max participants in any one experiment
    max_parts  = max(len(r.get("sessions", [])) for r, _ in valid)
    n_bars     = 2 + max_parts
    bar_width  = min(0.12, 0.8 / n_bars)
    x          = np.arange(n_groups)
    group_labels = []

    for gi, (r, lbl) in enumerate(valid):
        group_labels.append(lbl)
        sessions = r.get("sessions", [])
        fb_mean  = mean_trials(r["rewards_feedback"])
        bl_mean  = mean_trials(r["rewards_baseline"])

        # Baseline bar
        bl_val = tail_mean(bl_mean, window)
        offset = (-n_bars / 2 + 0.5) * bar_width
        ax.bar(x[gi] + offset, bl_val, width=bar_width,
               color=PALETTE["baseline"], alpha=0.9,
               label="Baseline" if gi == 0 else "_nolegend_")

        # Combined bar
        fb_val = tail_mean(fb_mean, window)
        offset = (-n_bars / 2 + 1.5) * bar_width
        ax.bar(x[gi] + offset, fb_val, width=bar_width,
               color=PALETTE["combined"], alpha=0.9,
               label="Combined feedback" if gi == 0 else "_nolegend_")

        # Per-participant bars — only this experiment's participants
        humans = [sh for sh in sessions if not is_llm(sh)]
        llms   = [sh for sh in sessions if is_llm(sh)]
        slot   = 2
        for sh in humans + llms:
            ind = r.get("rewards_individual", {}).get(sh)
            if ind is None:
                slot += 1
                continue
            val    = tail_mean(mean_trials(ind), window)
            color  = PALETTE["human"] if not is_llm(sh) else PALETTE["llm"]
            offset = (-n_bars / 2 + slot + 0.5) * bar_width
            lbl_h  = ("Human (individual)" if (not is_llm(sh) and gi == 0 and slot == 2)
                      else ("LLM (individual)" if (is_llm(sh) and gi == 0 and slot == len(humans) + 2)
                      else "_nolegend_"))
            ax.bar(x[gi] + offset, val, width=bar_width,
                   color=color, alpha=0.65, label=lbl_h)
            slot += 1

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=9)
    ax.set_ylabel("Mean reward (final 50 eps)")
    ax.set_title("Final performance — all experiments\n"
                 "(individuals shown within their own experiment only)",
                 fontsize=11, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    legend_elements = [
        Patch(facecolor=PALETTE["baseline"], label="Baseline (no feedback)"),
        Patch(facecolor=PALETTE["combined"], label="Combined feedback"),
        Patch(facecolor=PALETTE["human"],    alpha=0.65, label="Human (individual)"),
        Patch(facecolor=PALETTE["llm"],      alpha=0.65, label="LLM (individual)"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper right")


def plot_ce_bars(ax, all_results: list, labels: list):
    """Horizontal Ce bar chart: one group per experiment, bars = each participant."""
    valid = [(r, lbl) for r, lbl in zip(all_results, labels) if r is not None]
    if not valid:
        return

    n_exp      = len(valid)
    all_groups = []
    max_n      = 0

    for r, lbl in valid:
        sessions = r.get("sessions", [])
        ce_dict  = r.get("Ce_individual", {})
        entries  = [(sh, ce_dict.get(sh, float("nan"))) for sh in sessions]
        all_groups.append((lbl, entries))
        max_n = max(max_n, len(entries))

    bar_height = 0.7 / n_exp if n_exp > 0 else 0.7
    y          = np.arange(max_n)

    for gi, (lbl, entries) in enumerate(all_groups):
        ces    = [ce for _, ce in entries]
        names  = [sh for sh, _ in entries]
        offset = (gi - n_exp / 2 + 0.5) * bar_height
        colors = [PALETTE["llm"] if is_llm(sh) else PALETTE["human"] for sh in names]
        ax.barh(y[:len(ces)] + offset, ces, height=bar_height,
                color=colors, alpha=0.85, label=lbl)

    ax.set_yticks(y[:max_n])
    # Use session names from the last valid experiment as y-tick labels (approximate)
    last_sessions = [sh for sh, _ in all_groups[-1][1]]
    ax.set_yticklabels(last_sessions[:max_n], fontsize=7)
    ax.axvline(0.5, color="gray", linewidth=0.8, linestyle="--", label="Ce = 0.5")
    ax.axvline(0.75, color="red", linewidth=0.6, linestyle=":", label="Ce = 0.75 (reliable)")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Ce  (consistency estimate)")
    ax.set_title("Participant consistency (Ce) — all experiments", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Visualise all four experiment result pkl files in one figure."
    )
    p.add_argument("files", nargs="+", metavar="PKL",
                   help="Result pkl files in order: exp1 exp2 exp3 exp4")
    p.add_argument("--labels", nargs="+", default=None,
                   help="Experiment labels (default: Exp 1–4 names)")
    p.add_argument("--window",  type=int, default=20,
                   help="Smoothing window for learning curves (default: 20)")
    p.add_argument("--tail",    type=int, default=50,
                   help="Episodes to average for final-performance bars (default: 50)")
    p.add_argument("--save",    default=None,
                   help="Save figure to this path instead of displaying it")
    p.add_argument("--dpi",     type=int, default=150)
    args = p.parse_args()

    files  = args.files[:4]   # cap at 4
    labels = (args.labels or DEFAULT_LABELS)[:len(files)]
    # Pad if fewer than 4 labels provided
    while len(labels) < len(files):
        labels.append(f"Exp {len(labels)+1}")

    results = [load(f) for f in files]
    # Pad to 4 for indexing
    while len(results) < 4:
        results.append(None)
    while len(labels) < 4:
        labels.append("")

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 20))
    fig.patch.set_facecolor("#f8f8f8")
    gs  = gridspec.GridSpec(4, 2, figure=fig,
                            hspace=0.45, wspace=0.35,
                            height_ratios=[1, 1, 1.1, 1.1])

    ax_lc = [
        fig.add_subplot(gs[0, 0]),  # Exp 1
        fig.add_subplot(gs[0, 1]),  # Exp 2
        fig.add_subplot(gs[1, 0]),  # Exp 3
        fig.add_subplot(gs[1, 1]),  # Exp 4
    ]
    ax_bar = fig.add_subplot(gs[2, :])
    ax_ce  = fig.add_subplot(gs[3, :])

    # ── Learning curves ───────────────────────────────────────────────────────
    for i, (ax, r, lbl) in enumerate(zip(ax_lc, results, labels)):
        plot_learning_curve(ax, r, lbl, PALETTE["exp"][i], window=args.window)

    # ── Final performance bar ─────────────────────────────────────────────────
    plot_final_bar(ax_bar, results[:len(files)], labels[:len(files)], window=args.tail)

    # ── Ce bars ───────────────────────────────────────────────────────────────
    plot_ce_bars(ax_ce, results[:len(files)], labels[:len(files)])

    fig.suptitle("All Experiments — Offline Training Results", fontsize=14, fontweight="bold", y=0.995)

    # ── Output ────────────────────────────────────────────────────────────────
    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        fig.savefig(args.save, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved → {args.save}")
    else:
        matplotlib.use("TkAgg")
        plt.show()


if __name__ == "__main__":
    main()
