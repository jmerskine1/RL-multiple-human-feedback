#!/usr/bin/env python3
"""
analyse_results.py — Plot learning curves from offline_train.py output.

Shows all conditions on a single learning curve:
  • Baseline (no feedback)
  • Each participant individually
  • All participants combined

Usage
-----
  python analyse_results.py results.pkl
  python analyse_results.py results.pkl --window 20 --save plot.png
"""

import argparse
import pickle
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D


# ── Helpers ───────────────────────────────────────────────────────────────────

def rolling_mean(x, window):
    """
    Causal trailing rolling mean: each point is the average of the preceding
    `window` values (including itself).  No zero-padding — the window simply
    shrinks at the start of the series.  This avoids the edge artefacts
    (start spike / end dip) produced by np.convolve(..., mode="same").
    """
    x = np.array(x, dtype=float)
    if len(x) < 2:
        return x
    w = min(window, len(x))
    cs = np.cumsum(x)
    out = np.empty_like(x)
    out[:w] = cs[:w] / np.arange(1, w + 1)   # ramp-up: shrinking window
    out[w:] = (cs[w:] - cs[:-w]) / w          # full window
    return out


def rolling_std(x, window):
    x = np.array(x, dtype=float)
    result = np.full_like(x, 0.0)
    half = window // 2
    for i in range(len(x)):
        lo = max(0, i - half)
        hi = min(len(x), i + half + 1)
        result[i] = np.std(x[lo:hi])
    return result


def tail_mean(r, window):
    r = np.array(r, dtype=float)
    w = min(window, len(r))
    return float(np.mean(r[-w:]))


def _style_ax(ax):
    ax.set_facecolor("#111111")
    ax.tick_params(colors="white", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")


# ── Individual participant colours ────────────────────────────────────────────

# Distinct, readable colours that work on a dark background
_PARTICIPANT_PALETTE = [
    "#FF6B6B",  # coral red
    "#4ECDC4",  # teal
    "#A29BFE",  # lavender
    "#FD79A8",  # pink
    "#55EFC4",  # mint
    "#FDCB6E",  # peach
    "#74B9FF",  # sky blue
    "#E17055",  # orange
    "#00CEC9",  # dark teal
    "#B2BEC3",  # light grey
]

GOLD = "#FFD700"
BLUE = "#4663FF"


def participant_colour(idx: int) -> str:
    return _PARTICIPANT_PALETTE[idx % len(_PARTICIPANT_PALETTE)]


# ── Plotting ──────────────────────────────────────────────────────────────────

def _unpack_rewards(raw):
    """
    Accept either:
      - 1D array (n_episodes,)           — single-trial / old format
      - list / 2D array (n_trials, n_episodes) — multi-trial new format
    Returns (mean_curve, std_curve, all_trials) each shape (n_episodes,) /
    (n_trials, n_episodes).
    """
    arr = np.array(raw, dtype=float)
    if arr.ndim == 1:
        return arr, np.zeros_like(arr), arr[np.newaxis, :]
    else:
        return arr.mean(axis=0), arr.std(axis=0), arr


def plot_results(r: dict, window: int = 20, save_path: str = None):
    raw_fb  = r["rewards_feedback"]
    raw_bl  = r["rewards_baseline"]
    rewards_fb,  std_fb,  trials_fb  = _unpack_rewards(raw_fb)
    rewards_bl,  std_bl,  trials_bl  = _unpack_rewards(raw_bl)
    n_trials    = trials_fb.shape[0]
    rewards_ind = r.get("rewards_individual", {})   # {session: list-of-arrays or array}
    ce_ind      = r.get("Ce_individual", {})
    Ce          = np.array(r.get("Ce", []))
    sessions    = r.get("sessions", [])
    n_ep        = r.get("n_episodes", len(rewards_fb))
    episodes    = np.arange(1, len(rewards_fb) + 1)

    win = min(window, len(rewards_fb))

    # ── Layout ────────────────────────────────────────────────────────────────
    # 2 rows: top = learning curves (full width), bottom = bar + Ce
    fig = plt.figure(figsize=(15, 10), facecolor="#0a0a0a")
    gs  = gridspec.GridSpec(
        2, 2, figure=fig,
        hspace=0.38, wspace=0.28,
        left=0.07, right=0.97,
        top=0.88,  bottom=0.08,
    )
    ax_curve = fig.add_subplot(gs[0, :])
    ax_bar   = fig.add_subplot(gs[1, 0])
    ax_ce    = fig.add_subplot(gs[1, 1])

    for ax in (ax_curve, ax_bar, ax_ce):
        _style_ax(ax)

    # ── Learning curves ───────────────────────────────────────────────────────
    # With multiple trials: shading = cross-trial std (confidence, not noise).
    # With a single trial:  shading = rolling within-episode std (old behaviour).

    def _shade_std(mean_curve, std_curve, trials_arr, window):
        """Return (smoothed_mean, smoothed_shade) for plotting."""
        sm = rolling_mean(mean_curve, window)
        if trials_arr.shape[0] > 1:
            # Cross-trial std — smooth it too so shading isn't jagged
            sd = rolling_mean(std_curve, window)
        else:
            sd = rolling_std(mean_curve, window)
        return sm, sd

    # Individual participants — thin, semi-transparent
    for idx, (sh, rw_raw) in enumerate(rewards_ind.items()):
        rw_mean, rw_std, rw_trials = _unpack_rewards(rw_raw)
        col = participant_colour(idx)
        eps = np.arange(1, len(rw_mean) + 1)
        sm, sd = _shade_std(rw_mean, rw_std, rw_trials, window)
        if rw_trials.shape[0] == 1:
            ax_curve.plot(eps, rw_mean, color=col, alpha=0.10, linewidth=0.7)
        ax_curve.plot(eps, sm, color=col, alpha=0.80, linewidth=1.4,
                      label=f"{sh}  (Ce={ce_ind.get(sh, 0):.2f})")
        ax_curve.fill_between(eps, sm - sd, sm + sd, color=col, alpha=0.08)

    # Baseline — blue, medium weight
    sm_bl, sd_bl = _shade_std(rewards_bl, std_bl, trials_bl, window)
    if n_trials == 1:
        ax_curve.plot(episodes, rewards_bl, color=BLUE, alpha=0.12, linewidth=0.7)
    ax_curve.plot(episodes, sm_bl, color=BLUE, linewidth=2.2,
                  label=f"Baseline (no feedback)"
                        + (f"  n={n_trials}" if n_trials > 1 else ""),
                  zorder=5)
    ax_curve.fill_between(episodes, sm_bl - sd_bl, sm_bl + sd_bl,
                          color=BLUE, alpha=0.15)

    # Combined feedback — gold, thickest, on top
    sm_fb, sd_fb = _shade_std(rewards_fb, std_fb, trials_fb, window)
    if n_trials == 1:
        ax_curve.plot(episodes, rewards_fb, color=GOLD, alpha=0.12, linewidth=0.7)
    ax_curve.plot(episodes, sm_fb, color=GOLD, linewidth=2.8,
                  label=f"Combined (all participants)"
                        + (f"  n={n_trials}" if n_trials > 1 else ""),
                  zorder=6)
    ax_curve.fill_between(episodes, sm_fb - sd_fb, sm_fb + sd_fb,
                          color=GOLD, alpha=0.15)

    # Final mean dashes
    ax_curve.axhline(tail_mean(rewards_fb, win), color=GOLD, linewidth=0.8,
                     linestyle="--", alpha=0.45)
    ax_curve.axhline(tail_mean(rewards_bl, win), color=BLUE, linewidth=0.8,
                     linestyle="--", alpha=0.45)

    ax_curve.set_xlabel("Episode", color="white", fontsize=11)
    ax_curve.set_ylabel("Total Reward", color="white", fontsize=11)
    ax_curve.set_title("Learning Curves — all conditions", color=GOLD,
                       fontsize=13, pad=10)
    ax_curve.legend(
        facecolor="#111", edgecolor="#333",
        labelcolor="white", fontsize=9,
        loc="upper left",
    )

    # ── Final performance bar ─────────────────────────────────────────────────
    def _final_mean_std(trials_arr, win):
        """Mean and std of the last-win-episode average across trials."""
        per_trial = [np.mean(t[-win:]) for t in trials_arr]
        return float(np.mean(per_trial)), float(np.std(per_trial))

    bar_labels  = ["Baseline"]
    bar_means, bar_errs = zip(*[_final_mean_std(trials_bl, win)])
    bar_means  = list(bar_means)
    bar_errs   = list(bar_errs)
    bar_colours = [BLUE]

    for idx, sh in enumerate(sessions):
        if sh in rewards_ind:
            _, _, ind_trials = _unpack_rewards(rewards_ind[sh])
            m, e = _final_mean_std(ind_trials, win)
            bar_labels.append(sh)
            bar_means.append(m)
            bar_errs.append(e)
            bar_colours.append(participant_colour(idx))

    bar_labels.append("Combined")
    m, e = _final_mean_std(trials_fb, win)
    bar_means.append(m)
    bar_errs.append(e)
    bar_colours.append(GOLD)

    bar_values = bar_means
    x_pos = np.arange(len(bar_labels))
    bars  = ax_bar.bar(x_pos, bar_values, color=bar_colours,
                       width=0.6, edgecolor="#222",
                       yerr=bar_errs if n_trials > 1 else None,
                       capsize=4, error_kw={"ecolor": "white", "elinewidth": 1.2})

    max_val = max(abs(v) for v in bar_values) if bar_values else 1
    for bar, val in zip(bars, bar_values):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_val * 0.02,
            f"{val:.1f}",
            ha="center", va="bottom", color="white", fontsize=8,
        )

    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(bar_labels, rotation=30, ha="right", fontsize=8)
    ax_bar.set_title(f"Final Reward  (last {win} eps)",
                     color=GOLD, fontsize=11, pad=8)
    ax_bar.set_ylabel("Mean Reward", color="white", fontsize=10)

    # ── Ce per participant ────────────────────────────────────────────────────
    if len(Ce) > 0 and len(sessions) > 0:
        order      = np.argsort(-Ce)
        labels_ce  = [sessions[i] for i in order]
        values_ce  = [float(Ce[i]) for i in order]
        colours_ce = [participant_colour(list(sessions).index(sessions[i]))
                      for i in order]

        ce_bars = ax_ce.barh(range(len(labels_ce)), values_ce,
                             color=colours_ce, edgecolor="#222", height=0.55)
        ax_ce.set_yticks(range(len(labels_ce)))
        ax_ce.set_yticklabels(labels_ce, color="white", fontsize=9)
        ax_ce.set_xlim(0, 1)

        ax_ce.axvline(0.75, color=GOLD,    linewidth=0.9, linestyle="--",
                      alpha=0.6, label="Reliable (0.75)")
        ax_ce.axvline(0.55, color="orange",linewidth=0.9, linestyle="--",
                      alpha=0.6, label="Moderate (0.55)")
        ax_ce.axvline(0.50, color="#666",  linewidth=0.9, linestyle=":",
                      alpha=0.6, label="Random (0.50)")

        for bar, val in zip(ce_bars, values_ce):
            ax_ce.text(
                min(val + 0.02, 0.96),
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", color="white", fontsize=8,
            )

        ax_ce.set_title("Participant Consistency (Ce)\nfrom combined model",
                        color=GOLD, fontsize=11, pad=8)
        ax_ce.set_xlabel("Ce", color="white", fontsize=10)
        ax_ce.legend(facecolor="#111", edgecolor="#333",
                     labelcolor="white", fontsize=8, loc="lower right")
    else:
        ax_ce.text(0.5, 0.5, "No Ce data", transform=ax_ce.transAxes,
                   ha="center", va="center", color="#555", fontsize=12)

    # ── Title ─────────────────────────────────────────────────────────────────
    n_par = len(sessions)
    trial_str = f"  ·  {n_trials} trial{'s' if n_trials > 1 else ''}" if n_trials > 1 else ""
    fig.suptitle(
        f"Human Feedback RL  ·  {n_par} participant(s)  ·  {n_ep} episodes"
        f"{trial_str}  ·  smoothing window={window}",
        color=GOLD, fontsize=13, y=0.95, fontfamily="monospace",
    )

    out = save_path or "results/plots/learning_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Plot saved → {out}")
    plt.close(fig)


# ── Console report ────────────────────────────────────────────────────────────

def print_report(r: dict, window: int):
    sessions    = r.get("sessions", [])
    Ce          = np.array(r.get("Ce", []))
    ce_ind      = r.get("Ce_individual", {})
    rewards_fb  = np.array(r["rewards_feedback"])
    rewards_bl  = np.array(r["rewards_baseline"])
    rewards_ind = r.get("rewards_individual", {})

    win = min(window, len(rewards_fb))

    print("\n══ Participant Consistency ══════════════════════════════════════")
    print(f"  {'Session':<20} {'Ce (combined)':>14}  {'Ce (individual)':>16}")
    print("─" * 56)
    if len(Ce) > 0:
        for m in np.argsort(-Ce):
            sh      = sessions[m] if m < len(sessions) else str(m)
            ce_c    = float(Ce[m])
            ce_i    = ce_ind.get(sh, float("nan"))
            tag     = ("✓ reliable" if ce_c >= 0.75 else
                       "~ moderate" if ce_c >= 0.55 else
                       "✗ noisy")
            print(f"  {sh:<20} {ce_c:>14.3f}  {ce_i:>16.3f}   {tag}")
        print(f"\n  Combined mean Ce: {Ce.mean():.3f}   Std: {Ce.std():.3f}")

    print("\n══ Final performance ════════════════════════════════════════════")
    print(f"  {'Condition':<22} {'Mean reward (last ' + str(win) + ' eps)':>28}")
    print("─" * 56)
    print(f"  {'Baseline':<22} {tail_mean(rewards_bl, win):>28.2f}")
    for idx, sh in enumerate(sessions):
        if sh in rewards_ind:
            print(f"  {sh:<22} {tail_mean(rewards_ind[sh], win):>28.2f}")
    print(f"  {'Combined':<22} {tail_mean(rewards_fb, win):>28.2f}")

    delta = tail_mean(rewards_fb, win) - tail_mean(rewards_bl, win)
    pct   = (delta / abs(tail_mean(rewards_bl, win)) * 100
             if tail_mean(rewards_bl, win) != 0 else float("nan"))
    print(f"\n  Combined vs baseline:  Δ = {delta:+.2f}  ({pct:+.1f}%)\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Plot learning curves from offline_train.py results."
    )
    p.add_argument("results", help="Path to results.pkl")
    p.add_argument("--window", type=int, default=20,
                   help="Smoothing window size (default: 20)")
    p.add_argument("--save", default=None,
                   help="Output image path (default: results/plots/<input-stem>.png)")
    args = p.parse_args()

    try:
        with open(args.results, "rb") as f:
            r = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: file not found — {args.results}")
        sys.exit(1)

    # Default save path mirrors the input filename into results/plots/
    if args.save is None:
        import os
        stem = os.path.splitext(os.path.basename(args.results))[0]
        args.save = f"results/plots/{stem}.png"

    print_report(r, window=args.window)
    plot_results(r, window=args.window, save_path=args.save)


if __name__ == "__main__":
    main()
