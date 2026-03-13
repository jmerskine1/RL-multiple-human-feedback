#!/usr/bin/env python3
"""
compare_cest.py — Compare consistency-estimation ON vs OFF.

Loads two result files (with-Ce and without-Ce) and plots:
  • Top:    Combined + Baseline learning curves for both conditions
  • Bottom: Final-performance bar chart with error bars

Usage
-----
  python compare_cest.py results/results_all.pkl results/results_nocest_all.pkl
  python compare_cest.py with.pkl without.pkl --window 20 --save results/plots/cest_comparison.png
"""

import argparse
import pickle
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ── Helpers (copied from analyse_results.py so this script is self-contained) ─

GOLD   = "#FFD700"
BLUE   = "#4663FF"
GREEN  = "#00C897"   # Ce ON  — combined
ORANGE = "#FF7043"   # Ce OFF — combined


def rolling_mean(x, window):
    """Causal trailing rolling mean — no zero-padding edge artefacts."""
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
    """Return (mean_curve, std_curve, trials_2d)."""
    arr = np.array(raw, dtype=float)
    if arr.ndim == 1:
        return arr, np.zeros_like(arr), arr[np.newaxis, :]
    return arr.mean(axis=0), arr.std(axis=0), arr


def _style_ax(ax):
    ax.set_facecolor("#111111")
    ax.tick_params(colors="white", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")


def _plot_curve(ax, mean, std, trials, window, color, label, zorder=4):
    sm = rolling_mean(mean, window)
    sd = rolling_mean(std, window) if trials.shape[0] > 1 else np.zeros_like(sm)
    eps = np.arange(1, len(sm) + 1)
    ax.plot(eps, sm, color=color, linewidth=2.2, label=label, zorder=zorder)
    if trials.shape[0] > 1:
        ax.fill_between(eps, sm - sd, sm + sd, color=color, alpha=0.18)
    return float(tail_mean(sm, window))


def _final_mean_std(trials, win):
    per = [np.mean(t[-win:]) for t in trials]
    return float(np.mean(per)), float(np.std(per))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Compare Ce-ON vs Ce-OFF training results."
    )
    p.add_argument("with_cest",    help="Results pkl WITH Ce estimation")
    p.add_argument("without_cest", help="Results pkl WITHOUT Ce estimation (--no-cest)")
    p.add_argument("--window", type=int, default=20,
                   help="Smoothing window (default: 20)")
    p.add_argument("--save", default=None,
                   help="Output image path (default: results/plots/cest_comparison.png)")
    args = p.parse_args()

    for path in (args.with_cest, args.without_cest):
        if not os.path.exists(path):
            print(f"Error: file not found — {path}")
            sys.exit(1)

    with open(args.with_cest,    "rb") as f:
        r_on  = pickle.load(f)
    with open(args.without_cest, "rb") as f:
        r_off = pickle.load(f)

    win    = args.window
    n_ep   = r_on.get("n_episodes", 500)
    n_on   = r_on.get("n_trials",   1)
    n_off  = r_off.get("n_trials",  1)

    fb_mean_on,  fb_std_on,  fb_trials_on  = _unpack(r_on["rewards_feedback"])
    fb_mean_off, fb_std_off, fb_trials_off = _unpack(r_off["rewards_feedback"])
    bl_mean_on,  bl_std_on,  bl_trials_on  = _unpack(r_on["rewards_baseline"])
    bl_mean_off, bl_std_off, bl_trials_off = _unpack(r_off["rewards_baseline"])

    sessions_on  = r_on.get("sessions",  [])
    sessions_off = r_off.get("sessions", [])
    # Use whichever session list is available (should be identical)
    sessions = sessions_on or sessions_off

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 9), facecolor="#0a0a0a")
    gs  = gridspec.GridSpec(
        2, 2, figure=fig,
        hspace=0.42, wspace=0.28,
        left=0.07, right=0.97,
        top=0.88,  bottom=0.08,
    )
    ax_curve = fig.add_subplot(gs[0, :])
    ax_bar   = fig.add_subplot(gs[1, 0])
    ax_ce    = fig.add_subplot(gs[1, 1])

    for ax in (ax_curve, ax_bar, ax_ce):
        _style_ax(ax)

    # ── Learning curves ───────────────────────────────────────────────────────

    # Baseline — use the Ce-ON run (should be identical but average if both exist)
    bl_mean = (bl_mean_on + bl_mean_off) / 2
    bl_std  = (bl_std_on  + bl_std_off)  / 2
    bl_trials = np.vstack([bl_trials_on, bl_trials_off])
    _plot_curve(ax_curve, bl_mean, bl_std, bl_trials, win,
                BLUE, "Baseline (no feedback)", zorder=3)

    # Combined with Ce ON
    _plot_curve(ax_curve, fb_mean_on, fb_std_on, fb_trials_on, win,
                GREEN, f"Combined  —  Ce ON  (n={n_on})", zorder=6)

    # Combined with Ce OFF
    _plot_curve(ax_curve, fb_mean_off, fb_std_off, fb_trials_off, win,
                ORANGE, f"Combined  —  Ce OFF  (n={n_off})", zorder=5)

    ax_curve.set_xlabel("Episode", color="white", fontsize=11)
    ax_curve.set_ylabel("Total Reward", color="white", fontsize=11)
    ax_curve.set_title("Learning Curves — Consistency Estimation ON vs OFF",
                       color=GOLD, fontsize=13, pad=10)
    ax_curve.legend(
        facecolor="#111", edgecolor="#333",
        labelcolor="white", fontsize=10,
        loc="upper left",
    )

    # ── Final performance bar chart (grouped: one group per entity) ───────────
    # Each group = [Ce ON bar, Ce OFF bar] side by side.
    # X-axis has one label per group — avoids the label-overlap problem.
    rw_ind_on  = r_on.get("rewards_individual",  {})
    rw_ind_off = r_off.get("rewards_individual", {})

    bl_m,     bl_e     = _final_mean_std(bl_trials,     win)
    fb_on_m,  fb_on_e  = _final_mean_std(fb_trials_on,  win)
    fb_off_m, fb_off_e = _final_mean_std(fb_trials_off, win)

    group_labels = ["Baseline", "Combined"]
    on_means  = [bl_m,    fb_on_m]
    off_means = [bl_m,    fb_off_m]   # baseline is the same; keeps groups aligned
    on_errs   = [bl_e,    fb_on_e]
    off_errs  = [bl_e,    fb_off_e]

    for sh in sessions:
        if sh in rw_ind_on and sh in rw_ind_off:
            _, _, t_on  = _unpack(rw_ind_on[sh])
            _, _, t_off = _unpack(rw_ind_off[sh])
            m_on,  e_on  = _final_mean_std(t_on,  win)
            m_off, e_off = _final_mean_std(t_off, win)
            group_labels.append(sh)
            on_means.append(m_on);   off_means.append(m_off)
            on_errs.append(e_on);    off_errs.append(e_off)

    n_groups   = len(group_labels)
    x_pos      = np.arange(n_groups)
    bar_w      = 0.38
    n_trials_max = max(n_on, n_off)
    ek = {"ecolor": "white", "elinewidth": 1.1}

    bars_on  = ax_bar.bar(x_pos - bar_w / 2, on_means,  width=bar_w,
                          color=GREEN,  edgecolor="#222", label="Ce ON",
                          yerr=on_errs  if n_trials_max > 1 else None,
                          capsize=3, error_kw=ek)
    bars_off = ax_bar.bar(x_pos + bar_w / 2, off_means, width=bar_w,
                          color=ORANGE, edgecolor="#222", label="Ce OFF",
                          alpha=0.85,
                          yerr=off_errs if n_trials_max > 1 else None,
                          capsize=3, error_kw=ek)

    all_vals = on_means + off_means
    max_val  = max(abs(v) for v in all_vals) if all_vals else 1
    for bar, val in list(zip(bars_on, on_means)) + list(zip(bars_off, off_means)):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_val * 0.02,
            f"{val:.1f}",
            ha="center", va="bottom", color="white", fontsize=7,
        )

    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(group_labels, rotation=45, ha="right", fontsize=8)
    ax_bar.legend(facecolor="#111", edgecolor="#333",
                  labelcolor="white", fontsize=9, loc="upper left")
    ax_bar.set_title(f"Final Reward  (last {win} eps)",
                     color=GOLD, fontsize=11, pad=8)
    ax_bar.set_ylabel("Mean Reward", color="white", fontsize=10)

    # ── Ce values side-by-side ────────────────────────────────────────────────
    Ce_on  = np.array(r_on.get("Ce",  []))
    Ce_off = np.array(r_off.get("Ce", []))

    if len(Ce_on) > 0 and len(sessions_on) > 0:
        n_s   = len(sessions_on)
        y_pos = np.arange(n_s)
        height = 0.35

        ax_ce.barh(y_pos + height / 2, Ce_on,  height=height,
                   color=GREEN,  edgecolor="#222", label="Ce ON")
        ax_ce.barh(y_pos - height / 2, Ce_off, height=height,
                   color=ORANGE, edgecolor="#222", label="Ce OFF",
                   alpha=0.85)

        ax_ce.set_yticks(y_pos)
        ax_ce.set_yticklabels(sessions_on, color="white", fontsize=9)
        ax_ce.set_xlim(0, 1)
        ax_ce.axvline(0.75, color=GOLD,    linewidth=0.9, linestyle="--", alpha=0.6)
        ax_ce.axvline(0.50, color="#666",  linewidth=0.9, linestyle=":",  alpha=0.6)
        ax_ce.set_title("Participant Consistency (Ce)\nCe ON vs OFF",
                        color=GOLD, fontsize=11, pad=8)
        ax_ce.set_xlabel("Ce", color="white", fontsize=10)
        ax_ce.legend(facecolor="#111", edgecolor="#333",
                     labelcolor="white", fontsize=9, loc="lower right")
    else:
        ax_ce.text(0.5, 0.5, "No Ce data", transform=ax_ce.transAxes,
                   ha="center", va="center", color="#555", fontsize=12)

    # ── Title ─────────────────────────────────────────────────────────────────
    n_par = len(sessions)
    fig.suptitle(
        f"Consistency Estimation Ablation  ·  {n_par} participant(s)  "
        f"·  {n_ep} episodes  ·  window={win}",
        color=GOLD, fontsize=13, y=0.95, fontfamily="monospace",
    )

    out = args.save or "results/plots/cest_comparison.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Plot saved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
