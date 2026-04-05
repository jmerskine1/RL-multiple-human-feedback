#!/usr/bin/env python3
"""
compare_experiments.py — Compare two offline_train.py result files side by side.

Plots:
  Top:    Combined + Baseline learning curves for both experiments
  Bottom-left:  Final-performance bar chart (grouped pairs per LLM)
  Bottom-right: Ce per LLM for both experiments (paired horizontal bars)

Usage
-----
  # Compare Exp1 (count) vs Exp2 (entropy) with auto labels:
  python compare_experiments.py results/exp1_count_fixed.pkl results/exp2_entropy_fixed.pkl

  # With custom labels:
  python compare_experiments.py results/exp1_count_fixed.pkl results/exp2_entropy_fixed.pkl \
      --labels "Count" "Entropy"

  # Save to a specific path:
  python compare_experiments.py results/exp1.pkl results/exp2.pkl \
      --labels "Count" "Entropy" \
      --save results/plots/count_vs_entropy.png
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


# ── Style helpers ─────────────────────────────────────────────────────────────

GOLD   = "#FFD700"
BLUE   = "#4663FF"

# Two distinct colours for the two experiments
EXP_COLOURS = ["#00C897", "#FF7043"]   # teal / orange


def _style_ax(ax):
    ax.set_facecolor("#111111")
    ax.tick_params(colors="white", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")


def rolling_mean(x, window):
    x = np.array(x, dtype=float)
    if len(x) < 2:
        return x
    w = min(window, len(x))
    cs = np.cumsum(x)
    out = np.empty_like(x)
    out[:w] = cs[:w] / np.arange(1, w + 1)
    out[w:] = (cs[w:] - cs[:-w]) / w
    return out


def tail_mean(r, w):
    r = np.array(r, dtype=float)
    return float(np.mean(r[-min(w, len(r)):]))


def _unpack(raw):
    arr = np.array(raw, dtype=float)
    if arr.ndim == 1:
        return arr, np.zeros_like(arr), arr[np.newaxis, :]
    return arr.mean(axis=0), arr.std(axis=0), arr


def _final_mean_std(trials, win):
    per = [np.mean(t[-win:]) for t in trials]
    return float(np.mean(per)), float(np.std(per))


def _plot_curve(ax, mean, std, trials, window, color, label, zorder=4, alpha=1.0):
    sm = rolling_mean(mean, window)
    sd = rolling_mean(std, window) if trials.shape[0] > 1 else np.zeros_like(sm)
    eps = np.arange(1, len(sm) + 1)
    ax.plot(eps, sm, color=color, linewidth=2.2, label=label,
            zorder=zorder, alpha=alpha)
    if trials.shape[0] > 1:
        ax.fill_between(eps, sm - sd, sm + sd, color=color, alpha=0.15)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Compare two experiment result files side by side.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("result_a", help="First results .pkl file")
    p.add_argument("result_b", help="Second results .pkl file")
    p.add_argument("--labels", nargs=2, default=None, metavar=("LABEL_A", "LABEL_B"),
                   help="Display names for the two experiments (default: filenames)")
    p.add_argument("--window", type=int, default=20,
                   help="Smoothing window (default: 20)")
    p.add_argument("--save", default=None,
                   help="Output image path (default: results/plots/comparison.png)")
    args = p.parse_args()

    for path in (args.result_a, args.result_b):
        if not os.path.exists(path):
            print(f"Error: file not found — {path}")
            sys.exit(1)

    with open(args.result_a, "rb") as f:
        r_a = pickle.load(f)
    with open(args.result_b, "rb") as f:
        r_b = pickle.load(f)

    label_a = args.labels[0] if args.labels else os.path.splitext(os.path.basename(args.result_a))[0]
    label_b = args.labels[1] if args.labels else os.path.splitext(os.path.basename(args.result_b))[0]
    col_a, col_b = EXP_COLOURS

    win    = args.window
    n_ep   = r_a.get("n_episodes", 500)
    n_a    = r_a.get("n_trials", 1)
    n_b    = r_b.get("n_trials", 1)

    fb_mean_a, fb_std_a, fb_trials_a = _unpack(r_a["rewards_feedback"])
    fb_mean_b, fb_std_b, fb_trials_b = _unpack(r_b["rewards_feedback"])
    bl_mean_a, bl_std_a, bl_trials_a = _unpack(r_a["rewards_baseline"])
    bl_mean_b, bl_std_b, bl_trials_b = _unpack(r_b["rewards_baseline"])

    sessions_a = r_a.get("sessions", [])
    sessions_b = r_b.get("sessions", [])
    Ce_a       = np.array(r_a.get("Ce", []))
    Ce_b       = np.array(r_b.get("Ce", []))

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 9), facecolor="#0a0a0a")
    gs  = gridspec.GridSpec(
        2, 2, figure=fig,
        hspace=0.44, wspace=0.28,
        left=0.07, right=0.97,
        top=0.88,  bottom=0.08,
    )
    ax_curve = fig.add_subplot(gs[0, :])
    ax_bar   = fig.add_subplot(gs[1, 0])
    ax_ce    = fig.add_subplot(gs[1, 1])

    for ax in (ax_curve, ax_bar, ax_ce):
        _style_ax(ax)

    # ── Learning curves ───────────────────────────────────────────────────────
    # Shared baseline — average the two (should be near-identical)
    bl_mean_shared = (bl_mean_a + bl_mean_b) / 2
    bl_std_shared  = (bl_std_a  + bl_std_b)  / 2
    bl_trials_both = np.vstack([bl_trials_a, bl_trials_b])
    _plot_curve(ax_curve, bl_mean_shared, bl_std_shared, bl_trials_both, win,
                BLUE, "Baseline (no feedback)", zorder=3)

    _plot_curve(ax_curve, fb_mean_a, fb_std_a, fb_trials_a, win,
                col_a, f"{label_a}  (n={n_a})", zorder=6)
    _plot_curve(ax_curve, fb_mean_b, fb_std_b, fb_trials_b, win,
                col_b, f"{label_b}  (n={n_b})", zorder=5)

    ax_curve.set_xlabel("Episode", color="white", fontsize=11)
    ax_curve.set_ylabel("Total Reward", color="white", fontsize=11)
    ax_curve.set_title(f"Learning Curves — {label_a} vs {label_b}",
                       color=GOLD, fontsize=13, pad=10)
    ax_curve.legend(facecolor="#111", edgecolor="#333",
                    labelcolor="white", fontsize=10, loc="upper left")

    # ── Final performance bar chart ───────────────────────────────────────────
    # Grouped pairs: one group per LLM.
    # Sessions should correspond 1-to-1 by index (same 6 LLMs, different conditions).

    n_shared = min(len(sessions_a), len(sessions_b))
    rw_ind_a = r_a.get("rewards_individual", {})
    rw_ind_b = r_b.get("rewards_individual", {})

    bl_m, bl_e = _final_mean_std(bl_trials_both, win)
    fb_a_m, fb_a_e = _final_mean_std(fb_trials_a, win)
    fb_b_m, fb_b_e = _final_mean_std(fb_trials_b, win)

    # Build pairs: (short label, mean_a, err_a, mean_b, err_b)
    pairs = [("Baseline", bl_m, bl_e, bl_m, bl_e),
             ("Combined", fb_a_m, fb_a_e, fb_b_m, fb_b_e)]

    for i in range(n_shared):
        sh_a = sessions_a[i]
        sh_b = sessions_b[i]
        if sh_a in rw_ind_a and sh_b in rw_ind_b:
            _, _, t_a = _unpack(rw_ind_a[sh_a])
            _, _, t_b = _unpack(rw_ind_b[sh_b])
            m_a, e_a = _final_mean_std(t_a, win)
            m_b, e_b = _final_mean_std(t_b, win)
            # Use short session codes; strip hyphens for brevity if needed
            short = sh_a[:14] if len(sh_a) > 14 else sh_a
            pairs.append((short, m_a, e_a, m_b, e_b))

    group_labels = [lbl for lbl, *_ in pairs]
    means_a = [m for _, m, e, _, _ in pairs]
    errs_a  = [e for _, m, e, _, _ in pairs]
    means_b = [m for _, _, _, m, e in pairs]
    errs_b  = [e for _, _, _, m, e in pairs]

    n_groups = len(group_labels)
    x_pos    = np.arange(n_groups)
    bar_w    = 0.38
    n_max    = max(n_a, n_b)
    ek = {"ecolor": "white", "elinewidth": 1.1}

    bars_a = ax_bar.bar(x_pos - bar_w / 2, means_a, width=bar_w,
                        color=col_a, edgecolor="#222", label=label_a,
                        yerr=errs_a if n_max > 1 else None,
                        capsize=3, error_kw=ek)
    bars_b = ax_bar.bar(x_pos + bar_w / 2, means_b, width=bar_w,
                        color=col_b, edgecolor="#222", label=label_b,
                        alpha=0.85,
                        yerr=errs_b if n_max > 1 else None,
                        capsize=3, error_kw=ek)

    all_vals = means_a + means_b
    max_val  = max(abs(v) for v in all_vals) if all_vals else 1
    for bar, val in list(zip(bars_a, means_a)) + list(zip(bars_b, means_b)):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_val * 0.02,
            f"{val:.1f}",
            ha="center", va="bottom", color="white", fontsize=7,
        )

    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(group_labels, rotation=40, ha="right", fontsize=8)
    ax_bar.legend(facecolor="#111", edgecolor="#333",
                  labelcolor="white", fontsize=9, loc="upper left")
    ax_bar.set_title(f"Final Reward  (last {win} eps)",
                     color=GOLD, fontsize=11, pad=8)
    ax_bar.set_ylabel("Mean Reward", color="white", fontsize=10)

    # ── Ce per LLM — paired horizontal bars ──────────────────────────────────
    if len(Ce_a) > 0 and len(sessions_a) > 0:
        n_s    = min(len(sessions_a), len(Ce_a), len(sessions_b), len(Ce_b))
        y_pos  = np.arange(n_s)
        height = 0.35

        ax_ce.barh(y_pos + height / 2, Ce_a[:n_s], height=height,
                   color=col_a, edgecolor="#222", label=label_a)
        ax_ce.barh(y_pos - height / 2, Ce_b[:n_s], height=height,
                   color=col_b, edgecolor="#222", label=label_b, alpha=0.85)

        # Label each bar with its value
        for i in range(n_s):
            for val, offset in [(Ce_a[i], +height / 2), (Ce_b[i], -height / 2)]:
                ax_ce.text(min(float(val) + 0.02, 0.96), y_pos[i] + offset,
                           f"{val:.3f}", va="center", color="white", fontsize=7)

        ax_ce.set_yticks(y_pos)
        ax_ce.set_yticklabels(sessions_a[:n_s], color="white", fontsize=8)
        ax_ce.set_xlim(0, 1)
        ax_ce.axvline(0.75, color=GOLD,    linewidth=0.9, linestyle="--", alpha=0.6,
                      label="Reliable (0.75)")
        ax_ce.axvline(0.50, color="#666",  linewidth=0.9, linestyle=":",  alpha=0.6,
                      label="Random (0.50)")
        ax_ce.set_title(f"Participant Ce — {label_a} vs {label_b}",
                        color=GOLD, fontsize=11, pad=8)
        ax_ce.set_xlabel("Ce", color="white", fontsize=10)
        ax_ce.legend(facecolor="#111", edgecolor="#333",
                     labelcolor="white", fontsize=8, loc="lower right")
    else:
        ax_ce.text(0.5, 0.5, "No Ce data", transform=ax_ce.transAxes,
                   ha="center", va="center", color="#555", fontsize=12)

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\n{'═'*58}")
    print(f"  {'Session':<22}  {label_a:>12}   {label_b:>12}   Δ")
    print(f"─{'─'*57}")
    for lbl, m_a, _, m_b, _ in pairs:
        delta = m_b - m_a
        print(f"  {lbl:<22}  {m_a:>12.2f}   {m_b:>12.2f}   {delta:>+.2f}")
    print(f"{'═'*58}\n")

    if len(Ce_a) > 0:
        print(f"  Ce  ({label_a})  mean={Ce_a.mean():.3f}   ({label_b})  mean={Ce_b.mean():.3f}\n")

    # ── Title + save ──────────────────────────────────────────────────────────
    fig.suptitle(
        f"{label_a} vs {label_b}  ·  {n_ep} episodes  ·  "
        f"n={n_a}/{n_b} trials  ·  window={win}",
        color=GOLD, fontsize=13, y=0.95, fontfamily="monospace",
    )

    out = args.save or "results/plots/comparison.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Plot saved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
