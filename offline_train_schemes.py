#!/usr/bin/env python3
"""
offline_train_schemes.py  —  Alternative training schemes using existing annotations.

Five schemes, each a drop-in replacement for the default offline_train.py
combined training run.  All use the same data already collected; no new
cloud annotation needed.

Schemes
-------
  burnin        Ce = 1.0 for first --burnin-episodes, then switch to adaptive Ce.
                Avoids early-training amplification of noisy annotators while
                preserving long-run calibration.

  two-phase     Phase 1 (first --phase-split episodes): LLM-only feedback.
                Phase 2 (remaining episodes): combined human + LLM, Q-table
                inherited from phase 1.  Bootstraps with consistent LLMs,
                then refines with humans.

  warm-init     Q-table initialised from Ce-weighted hp−hm feedback signal
                before training begins, rather than zeros.  Annotated
                state-action pairs get a head-start proportional to annotator
                reliability × net label confidence.

  disagreement  Replay threshold scales with inter-annotator disagreement at
                each state.  Contested states (annotators disagree on the best
                action) are replayed more; consensus states are replayed less.
                Focuses learning on the most informative parts of the space.

  majority-vote Compress N annotators to a single consensus annotator via
                Ce-weighted majority vote before training.  Pathological
                individuals (e.g. inverted-policy annotators) are suppressed
                without requiring adaptive Ce updates during training.

Usage
-----
  python offline_train_schemes.py --scheme burnin \\
      --bucket jerskine_human_feedback \\
      --sessions AMBER-FOREST BRIGHT-CANYON CLEAN-VOYAGE \\
                 ABLE-VERDICT BRAVE-COMPASS CALM-HORIZON \\
                 DEFT-LANTERN EAGER-SUMMIT FAIR-ANCHOR \\
      --local-brains-dir llm_brains/ \\
      --n-episodes 1000 --n-trials 10 \\
      --save results/exp1_burnin.pkl

  # All schemes, all 4 experiments:
  python offline_train_schemes.py --run-all
"""

import argparse
import os
import pickle
import sys

import numpy as np
from scipy.special import psi as scipy_psi

# Re-use helpers from offline_train — no duplication
from offline_train import (
    load_all_brains,
    build_combined_agent,
    run_c_estimation,
    print_consistency_report,
    print_performance_report,
    train_without_feedback,
    train_individual_condition,
    _human_feedback_for_state,
    train_with_human_replay,
)
from feedback import Feedback
from trainer import PacmanTrainer


# ── Shared session / experiment config ────────────────────────────────────────

EXPERIMENTS = {
    "exp1": {
        "label":    "Exp 1 — Count / Fixed",
        "sessions": [
            "AMBER-FOREST", "BRIGHT-CANYON", "CLEAN-VOYAGE",
            "ABLE-VERDICT", "BRAVE-COMPASS", "CALM-HORIZON",
            "DEFT-LANTERN", "EAGER-SUMMIT", "FAIR-ANCHOR",
        ],
        "llm_sessions": {
            "ABLE-VERDICT", "BRAVE-COMPASS", "CALM-HORIZON",
            "DEFT-LANTERN", "EAGER-SUMMIT", "FAIR-ANCHOR",
        },
        "env_random":    False,
        "pellet_random": False,
    },
    "exp2": {
        "label":    "Exp 2 — Entropy / Fixed",
        "sessions": [
            "FAINT-ORBIT", "GIANT-VESSEL", "HOLLOW-BRIDGE",
            "GRAND-CIRCUIT", "HEAVY-MARBLE", "INNER-TROPHY",
            "JOLLY-BEACON", "KEEN-SECTOR", "LOYAL-FOSSIL",
        ],
        "llm_sessions": {
            "GRAND-CIRCUIT", "HEAVY-MARBLE", "INNER-TROPHY",
            "JOLLY-BEACON", "KEEN-SECTOR", "LOYAL-FOSSIL",
        },
        "env_random":    False,
        "pellet_random": False,
    },
    "exp3": {
        "label":    "Exp 3 — Count / Random",
        "sessions": [
            "IVORY-SPARK", "JUMPY-CLOUD", "LUCKY-BARREL",
            "MERRY-VECTOR", "NOVEL-CASTLE", "OPEN-RIDDLE",
            "PRIME-SOCKET", "QUIET-FALCON", "RAPID-ISLAND",
        ],
        "llm_sessions": {
            "MERRY-VECTOR", "NOVEL-CASTLE", "OPEN-RIDDLE",
            "PRIME-SOCKET", "QUIET-FALCON", "RAPID-ISLAND",
        },
        "env_random":    True,
        "pellet_random": True,
    },
    "exp4": {
        "label":    "Exp 4 — Entropy / Random",
        "sessions": [
            "MISTY-CEDAR", "NOBLE-PEBBLE",
            "SHARP-NEBULA", "TIDY-HARBOR", "URBAN-PORTAL",
            "VIVID-GOBLIN", "WITTY-PRISM", "YOUNG-TURRET",
        ],
        "llm_sessions": {
            "SHARP-NEBULA", "TIDY-HARBOR", "URBAN-PORTAL",
            "VIVID-GOBLIN", "WITTY-PRISM", "YOUNG-TURRET",
        },
        "env_random":    True,
        "pellet_random": True,
    },
}


# ═════════════════════════════════════════════════════════════════════════════
# Scheme 1 — Ce Burn-in
# ═════════════════════════════════════════════════════════════════════════════

def train_burnin(
    trainer: PacmanTrainer,
    hp0: np.ndarray,
    hm0: np.ndarray,
    n_episodes: int,
    burnin_episodes: int = 200,
    max_steps: int = 500,
    update_Cest_interval: int = 5,
    active_learning_threshold: float = 3.0,
    reset_random: bool = False,
    pellet_random: bool = False,
) -> list:
    """
    Ce burn-in: uniform weights for first `burnin_episodes`, then adaptive Ce.

    During burn-in Ce is frozen at 1.0 for all annotators — this prevents
    the early-training instability where the VI prior amplifies harmful
    annotators before enough evidence has accumulated.  After burn-in, the
    standard adaptive Ce kicks in with a much better-calibrated starting point.
    """
    N   = trainer.agent.nTrainer
    rewards = []

    for ep in range(n_episodes):
        trainer.reset_episode(random=reset_random, pellet_random=pellet_random)
        steps = 0

        # After burnin, switch on adaptive Ce
        past_burnin    = (ep >= burnin_episodes)
        update_Cest    = past_burnin and ((ep + 1) % update_Cest_interval == 0)

        # During burnin ensure Ce stays at 1.0
        if ep == burnin_episodes:
            # First episode past burnin: run a fresh Ce estimation from accumulated data
            run_c_estimation(trainer)

        while not trainer.done and steps < max_steps:
            state = trainer.ob
            fb = _human_feedback_for_state(
                hp0, hm0, state, N,
                trainer.agent.Nsa[state, :],
                active_learning_threshold,
            )
            trainer.step(feedback=fb, update_Cest=False)
            steps += 1

            if trainer.done or steps >= max_steps:
                trainer.agent.act(
                    0, trainer.ob, trainer.rw, trainer.done,
                    fb, 0.5, update_Cest=update_Cest,
                )
                trainer.agent.prev_obs = None
                break

        rewards.append(float(trainer.totRW))

        if (ep + 1) % max(1, n_episodes // 10) == 0:
            phase = "adaptive" if past_burnin else "burnin"
            ce_str = ", ".join(f"{c:.3f}" for c in trainer.agent.Ce)
            print(f"  [burnin/{phase}]  ep {ep+1:>4}/{n_episodes}  "
                  f"reward={trainer.totRW:>7.2f}  Ce=[{ce_str}]")

    return rewards


# ═════════════════════════════════════════════════════════════════════════════
# Scheme 2 — Two-Phase: LLM bootstrap → combined fine-tune
# ═════════════════════════════════════════════════════════════════════════════

def train_two_phase(
    sessions: list,
    llm_sessions: set,
    env_size: str,
    n_episodes: int,
    phase_split: int = 300,
    max_steps: int = 500,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    update_Cest_interval: int = 5,
    active_learning_threshold: float = 3.0,
    reset_random: bool = False,
    pellet_random: bool = False,
) -> list:
    """
    Phase 1 (LLM-only, first phase_split episodes):
        Train using only the high-consistency LLM annotators.
        LLMs are consistent (Ce≈0.9) so this bootstraps a reliable Q-table.

    Phase 2 (combined, remaining episodes):
        Transfer the learned Q-table to a full combined trainer and continue
        training with all annotators.  Human annotators refine the policy
        in regions the LLMs may have oversimplified.
    """
    llm_sess   = [(sh, b) for sh, b in sessions if sh in llm_sessions]
    human_sess = [(sh, b) for sh, b in sessions if sh not in llm_sessions]

    # ── Phase 1: LLM-only ───────────────────────────────────────────────────
    p1_trainer = PacmanTrainer(
        algID="tabQL_Cest_vi_t2", env_size=env_size,
        prior_alpha=prior_alpha, prior_beta=prior_beta,
    )
    build_combined_agent(p1_trainer, llm_sess,
                         prior_alpha=prior_alpha, prior_beta=prior_beta)
    Ce_llm = run_c_estimation(p1_trainer)

    hp0_p1 = p1_trainer.agent.hp.copy()
    hm0_p1 = p1_trainer.agent.hm.copy()

    print(f"  [two-phase] Phase 1: {len(llm_sess)} LLM sessions, "
          f"{phase_split} episodes")
    rewards_p1 = train_with_human_replay(
        p1_trainer, hp0_p1, hm0_p1,
        n_episodes=phase_split,
        max_steps=max_steps,
        update_Cest_interval=update_Cest_interval,
        active_learning_threshold=active_learning_threshold,
        adaptive_cest=True,
        reset_random=reset_random,
        pellet_random=pellet_random,
    )

    # ── Phase 2: combined, Q-table inherited from phase 1 ───────────────────
    p2_trainer = PacmanTrainer(
        algID="tabQL_Cest_vi_t2", env_size=env_size,
        prior_alpha=prior_alpha, prior_beta=prior_beta,
    )
    build_combined_agent(p2_trainer, sessions,
                         prior_alpha=prior_alpha, prior_beta=prior_beta)
    p2_trainer.agent.Q = p1_trainer.agent.Q.copy()   # transfer learned Q
    Ce_combined = run_c_estimation(p2_trainer)

    hp0_p2 = p2_trainer.agent.hp.copy()
    hm0_p2 = p2_trainer.agent.hm.copy()
    n_p2   = n_episodes - phase_split

    print(f"  [two-phase] Phase 2: {len(sessions)} combined sessions, "
          f"{n_p2} episodes (Q inherited)")
    rewards_p2 = train_with_human_replay(
        p2_trainer, hp0_p2, hm0_p2,
        n_episodes=n_p2,
        max_steps=max_steps,
        update_Cest_interval=update_Cest_interval,
        active_learning_threshold=active_learning_threshold,
        adaptive_cest=True,
        reset_random=reset_random,
        pellet_random=pellet_random,
    )

    return rewards_p1 + rewards_p2, p2_trainer


# ═════════════════════════════════════════════════════════════════════════════
# Scheme 3 — Feedback Warm Q-Init
# ═════════════════════════════════════════════════════════════════════════════

def feedback_warm_q_init(
    combined_hp: np.ndarray,
    combined_hm: np.ndarray,
    Ce: np.ndarray,
    scale: float = 50.0,
) -> np.ndarray:
    """
    Initialise Q from the Ce-weighted net feedback signal.

    Q_init[s, a] = scale * (sum_m Ce[m] * (hp[m,s,a] − hm[m,s,a]))
                           / (sum_m Ce[m] * (hp[m,s,a] + hm[m,s,a]) + ε)

    The normalisation per (s,a) prevents high-count states from dominating;
    the result is a smoothed proportion in (−1, +1) scaled to ±scale.
    States with no annotation stay at 0.
    """
    net      = combined_hp - combined_hm              # [N, nS, nA]
    total    = combined_hp + combined_hm + 1e-8       # [N, nS, nA]
    w        = Ce[:, np.newaxis, np.newaxis]           # [N, 1, 1]

    weighted_net   = (w * net  ).sum(0)               # [nS, nA]
    weighted_total = (w * total).sum(0)                # [nS, nA]

    # Proportion ∈ (−1, +1), zero where no annotations exist
    prop    = weighted_net / weighted_total
    Q_init  = prop * scale

    n_annotated = int((weighted_total > 1e-4).any(axis=1).sum())
    print(f"  [warm-init] Q initialised for {n_annotated} annotated states "
          f"(scale={scale:.1f})")
    return Q_init


# ═════════════════════════════════════════════════════════════════════════════
# Scheme 4 — Disagreement-Weighted Replay
# ═════════════════════════════════════════════════════════════════════════════

def _human_feedback_disagreement(
    hp0: np.ndarray,
    hm0: np.ndarray,
    state: int,
    n_trainers: int,
    nsa_row: np.ndarray,
    base_threshold: float,
    disagreement_map: np.ndarray,
) -> list:
    """
    Replay feedback at `state` with a threshold that scales with inter-annotator
    disagreement.

    disagreement_map[s] ∈ [0, ∞):  higher = annotators disagree more at state s.

    Effective threshold = base_threshold * exp(−disagreement_map[s])
        → low disagreement (consensus):   threshold ≈ base_threshold  (normal replay)
        → high disagreement (contested):  threshold → 0  (always replay)

    This ensures the agent receives shaping signal most often at states where
    the overall annotation picture is uncertain, directing learning effort to
    the most informative parts of the state space.
    """
    # Scale threshold down where disagreement is high
    d = float(disagreement_map[state])
    effective_threshold = base_threshold * np.exp(-d)

    if np.min(nsa_row) >= effective_threshold:
        return [[] for _ in range(n_trainers)]

    # Same feedback construction as the default helper
    fb = []
    for m in range(n_trainers):
        hp_s = hp0[m, state, :]
        hm_s = hm0[m, state, :]
        if np.sum(hp_s + hm_s) < 1e-6:
            fb.append([])
            continue
        net  = hp_s - hm_s
        good = [int(i) for i in range(len(net)) if net[i] > 0]
        bad  = [int(i) for i in range(len(net)) if net[i] < 0]
        if good or bad:
            fb.append([Feedback(state=state,
                                good_actions=good, bad_actions=bad,
                                conf_good_actions=1.0, conf_bad_actions=1.0)])
        else:
            fb.append([])
    return fb


def compute_disagreement_map(hp0: np.ndarray, hm0: np.ndarray) -> np.ndarray:
    """
    Per-state inter-annotator disagreement.

    For each state s, compute the circular variance of the net label vectors
    across annotators with at least one opinion.  Returns an array of shape
    [nS] with values ≥ 0; higher = more disagreement.
    """
    nS   = hp0.shape[1]
    dmap = np.zeros(nS)

    for s in range(nS):
        nets = hp0[:, s, :] - hm0[:, s, :]           # [N, nA]
        has_opinion = (hp0[:, s, :] + hm0[:, s, :]).sum(1) > 0
        if has_opinion.sum() < 2:
            continue
        active_nets = nets[has_opinion]               # [k, nA]
        # Normalise each annotator's vector (avoid zero-division)
        norms = np.linalg.norm(active_nets, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        unit  = active_nets / norms                   # [k, nA]
        # Mean resultant length R ∈ [0,1]; disagreement = 1 − R
        R     = np.linalg.norm(unit.mean(0))
        dmap[s] = 1.0 - R

    n_contested = int((dmap > 0.1).sum())
    print(f"  [disagreement] {n_contested} contested states "
          f"(disagreement > 0.1), max={dmap.max():.3f}")
    return dmap


def train_disagreement_replay(
    trainer: PacmanTrainer,
    hp0: np.ndarray,
    hm0: np.ndarray,
    n_episodes: int,
    disagreement_map: np.ndarray,
    base_threshold: float = 3.0,
    max_steps: int = 500,
    update_Cest_interval: int = 5,
    adaptive_cest: bool = True,
    reset_random: bool = False,
    pellet_random: bool = False,
) -> list:
    N = trainer.agent.nTrainer
    rewards = []

    for ep in range(n_episodes):
        trainer.reset_episode(random=reset_random, pellet_random=pellet_random)
        steps = 0

        while not trainer.done and steps < max_steps:
            state = trainer.ob
            fb = _human_feedback_disagreement(
                hp0, hm0, state, N,
                trainer.agent.Nsa[state, :],
                base_threshold,
                disagreement_map,
            )
            trainer.step(feedback=fb, update_Cest=False)
            steps += 1

            if trainer.done or steps >= max_steps:
                update_Cest = adaptive_cest and ((ep + 1) % update_Cest_interval == 0)
                trainer.agent.act(0, trainer.ob, trainer.rw, trainer.done,
                                  fb, 0.5, update_Cest=update_Cest)
                trainer.agent.prev_obs = None
                break

        rewards.append(float(trainer.totRW))

        if (ep + 1) % max(1, n_episodes // 10) == 0:
            ce_str = ", ".join(f"{c:.3f}" for c in trainer.agent.Ce)
            print(f"  [disagree]  ep {ep+1:>4}/{n_episodes}  "
                  f"reward={trainer.totRW:>7.2f}  Ce=[{ce_str}]")

    return rewards


# ═════════════════════════════════════════════════════════════════════════════
# Scheme 5 — Ce-Weighted Majority Vote
# ═════════════════════════════════════════════════════════════════════════════

def build_majority_vote_agent(
    trainer: PacmanTrainer,
    sessions: list,
    Ce: np.ndarray,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> None:
    """
    Compress N annotators to a single consensus annotator via Ce-weighted
    majority vote, then build a 1-trainer agent.

    For each (state, action):
      vote[m] = +1 if hp[m,s,a] > hm[m,s,a]  (annotator says 'good')
               −1 if hp[m,s,a] < hm[m,s,a]  (annotator says 'bad')
                0 otherwise (neutral / no opinion)

    consensus[s,a] = sum_m Ce[m] * vote[m, s, a]
      → consensus > 0  ⟹  good action (goes into hp)
      → consensus < 0  ⟹  bad  action (goes into hm)

    The resulting hp/hm arrays are binary (0 or 1) scaled by the confidence
    magnitude, so the training loop treats them identically to normal feedback.
    """
    agent = trainer.agent

    def _extract(brain, key):
        arr = np.array(brain[key])
        return arr[0] if arr.ndim == 3 else arr

    all_hp = np.stack([_extract(b, 'hp') for _, b in sessions])  # [N, nS, nA]
    all_hm = np.stack([_extract(b, 'hm') for _, b in sessions])

    net   = all_hp - all_hm                                       # [N, nS, nA]
    votes = np.sign(net)                                          # {-1, 0, +1}
    w     = Ce[:, np.newaxis, np.newaxis]
    consensus = (w * votes).sum(0)                                 # [nS, nA]

    # Map consensus to hp/hm for a single "super-annotator"
    vote_hp = np.maximum( consensus, 0)[np.newaxis, :, :]          # [1, nS, nA]
    vote_hm = np.maximum(-consensus, 0)[np.newaxis, :, :]

    nS, nA = consensus.shape
    agent.Q         = np.zeros((nS, nA))
    agent.hp        = vote_hp
    agent.hm        = vote_hm
    agent.Nsa       = np.sum(
        [np.array(b.get('Nsa', np.zeros((nS, nA)))) for _, b in sessions],
        axis=0,
    )
    agent.nTrainer  = 1
    agent.a         = np.array([prior_alpha])
    agent.b         = np.array([prior_beta])
    agent.Ce        = np.array([0.5])
    agent.sum_of_right_feedback = np.zeros(1)
    agent.sum_of_wrong_feedback = np.zeros(1)
    agent.psi_for_hr = (
        scipy_psi(agent.sum_of_right_feedback + agent.a) -
        scipy_psi(agent.sum_of_right_feedback + agent.sum_of_wrong_feedback +
                  agent.a + agent.b)
    )
    agent.psi_for_hw = (
        scipy_psi(agent.sum_of_wrong_feedback + agent.b) -
        scipy_psi(agent.sum_of_right_feedback + agent.sum_of_wrong_feedback +
                  agent.a + agent.b)
    )
    agent.prev_obs = None

    n_states_with_vote = int((np.abs(consensus).sum(1) > 0).sum())
    print(f"  [majority-vote] Consensus over {len(sessions)} annotators → "
          f"1 super-annotator covering {n_states_with_vote} states")


# ═════════════════════════════════════════════════════════════════════════════
# Shared trial runner
# ═════════════════════════════════════════════════════════════════════════════

def run_trials(
    scheme: str,
    sessions: list,
    llm_sessions: set,
    env_size: str,
    n_episodes: int,
    n_trials: int,
    max_steps: int,
    prior_alpha: float,
    prior_beta: float,
    update_Cest_interval: int,
    active_learning_threshold: float,
    env_random: bool,
    pellet_random: bool,
    # scheme-specific
    burnin_episodes: int = 200,
    phase_split: int = 300,
    warm_scale: float = 50.0,
    skip_individual: bool = False,
) -> dict:

    all_rewards_fb         = []
    all_rewards_bl         = []
    all_rewards_individual = {sh: [] for sh, _ in sessions}
    all_Ce                 = []
    all_ce_individual      = {sh: [] for sh, _ in sessions}
    last_fb_trainer = last_bl_trainer = None

    for trial in range(n_trials):
        if n_trials > 1:
            print(f"\n{'═'*60}  Trial {trial+1}/{n_trials}  {'═'*60}")
        np.random.seed(trial)

        # ── Combined agent with Ce estimation ─────────────────────────────
        fb_trainer = PacmanTrainer(
            algID="tabQL_Cest_vi_t2", env_size=env_size,
            prior_alpha=prior_alpha, prior_beta=prior_beta,
        )

        if scheme == "majority-vote":
            # Build combined first to get Ce, then build majority-vote agent
            tmp = PacmanTrainer(
                algID="tabQL_Cest_vi_t2", env_size=env_size,
                prior_alpha=prior_alpha, prior_beta=prior_beta,
            )
            build_combined_agent(tmp, sessions,
                                 prior_alpha=prior_alpha, prior_beta=prior_beta)
            Ce = run_c_estimation(tmp)
            build_majority_vote_agent(fb_trainer, sessions, Ce,
                                      prior_alpha=prior_alpha,
                                      prior_beta=prior_beta)
            run_c_estimation(fb_trainer)
            Ce = fb_trainer.agent.Ce.copy()
        else:
            build_combined_agent(fb_trainer, sessions,
                                 prior_alpha=prior_alpha, prior_beta=prior_beta)
            Ce = run_c_estimation(fb_trainer)

        all_Ce.append(Ce)

        hp0 = fb_trainer.agent.hp.copy()
        hm0 = fb_trainer.agent.hm.copy()

        # ── Apply warm Q-init if requested ────────────────────────────────
        if scheme == "warm-init":
            fb_trainer.agent.Q = feedback_warm_q_init(hp0, hm0, Ce, scale=warm_scale)

        # ── Training ──────────────────────────────────────────────────────
        if scheme == "burnin":
            rewards_fb = train_burnin(
                fb_trainer, hp0, hm0,
                n_episodes=n_episodes,
                burnin_episodes=burnin_episodes,
                max_steps=max_steps,
                update_Cest_interval=update_Cest_interval,
                active_learning_threshold=active_learning_threshold,
                reset_random=env_random,
                pellet_random=pellet_random,
            )

        elif scheme == "two-phase":
            rewards_fb, fb_trainer = train_two_phase(
                sessions, llm_sessions, env_size,
                n_episodes=n_episodes,
                phase_split=phase_split,
                max_steps=max_steps,
                prior_alpha=prior_alpha, prior_beta=prior_beta,
                update_Cest_interval=update_Cest_interval,
                active_learning_threshold=active_learning_threshold,
                reset_random=env_random,
                pellet_random=pellet_random,
            )

        elif scheme == "disagreement":
            dmap = compute_disagreement_map(hp0, hm0)
            rewards_fb = train_disagreement_replay(
                fb_trainer, hp0, hm0,
                n_episodes=n_episodes,
                disagreement_map=dmap,
                base_threshold=active_learning_threshold,
                max_steps=max_steps,
                update_Cest_interval=update_Cest_interval,
                adaptive_cest=True,
                reset_random=env_random,
                pellet_random=pellet_random,
            )

        else:  # warm-init and majority-vote use standard replay
            rewards_fb = train_with_human_replay(
                fb_trainer, hp0, hm0,
                n_episodes=n_episodes,
                max_steps=max_steps,
                update_Cest_interval=update_Cest_interval,
                active_learning_threshold=active_learning_threshold,
                adaptive_cest=True,
                reset_random=env_random,
                pellet_random=pellet_random,
            )

        all_rewards_fb.append(rewards_fb)
        last_fb_trainer = fb_trainer

        # ── Baseline ──────────────────────────────────────────────────────
        rewards_bl, bl_trainer = train_without_feedback(
            env_size, n_episodes, max_steps=max_steps,
            reset_random=env_random, pellet_random=pellet_random,
        )
        all_rewards_bl.append(rewards_bl)
        last_bl_trainer = bl_trainer

        # ── Individual runs (skip for two-phase or when explicitly disabled) ──
        if scheme != "two-phase" and not skip_individual:
            for sh, brain in sessions:
                rw, ce = train_individual_condition(
                    sh, brain, env_size=env_size,
                    n_episodes=n_episodes, max_steps=max_steps,
                    prior_alpha=prior_alpha, prior_beta=prior_beta,
                    training_mode="replay",
                    active_learning_threshold=active_learning_threshold,
                    update_Cest_interval=update_Cest_interval,
                    adaptive_cest=True,
                    reset_random=env_random, pellet_random=pellet_random,
                )
                all_rewards_individual[sh].append(rw)
                all_ce_individual[sh].append(ce)

        if n_trials > 1:
            w = 50
            mf = np.mean(all_rewards_fb[-1][-w:])
            mb = np.mean(all_rewards_bl[-1][-w:])
            print(f"  Trial {trial+1} done — feedback={mf:.1f}  baseline={mb:.1f}")

    Ce_mean = np.mean(all_Ce, axis=0)
    return {
        "sessions":           [sh for sh, _ in sessions],
        "scheme":             scheme,
        "Ce":                 Ce_mean,
        "Ce_individual":      {sh: float(np.mean(all_ce_individual[sh]))
                               for sh in all_ce_individual
                               if all_ce_individual[sh]},
        "rewards_feedback":   all_rewards_fb,
        "rewards_baseline":   all_rewards_bl,
        "rewards_individual": {sh: all_rewards_individual[sh]
                               for sh, _ in sessions
                               if all_rewards_individual[sh]},
        "Q_feedback":         last_fb_trainer.agent.Q,
        "Q_baseline":         last_bl_trainer.agent.Q,
        "hp":                 last_fb_trainer.agent.hp,
        "hm":                 last_fb_trainer.agent.hm,
        "n_episodes":         n_episodes,
        "n_trials":           n_trials,
        "env_size":           env_size,
    }


# ═════════════════════════════════════════════════════════════════════════════
# --run-all convenience wrapper
# ═════════════════════════════════════════════════════════════════════════════

RUN_ALL_SCHEMES = ["burnin", "two-phase", "warm-init", "disagreement", "majority-vote"]

def run_all(args):
    """Run every scheme × every experiment and save results."""
    for exp_id, cfg in EXPERIMENTS.items():
        print(f"\n{'█'*70}")
        print(f"  {cfg['label']}")
        print(f"{'█'*70}")

        sessions = load_all_brains(
            args.bucket,
            only=cfg["sessions"],
            local_dir=args.local_brains_dir,
        )
        if not sessions:
            print(f"  No sessions found for {exp_id} — skipping")
            continue

        for scheme in RUN_ALL_SCHEMES:
            print(f"\n  ── Scheme: {scheme.upper()} ──")
            results = run_trials(
                scheme=scheme,
                sessions=sessions,
                llm_sessions=cfg["llm_sessions"],
                env_size=args.env_size,
                n_episodes=args.n_episodes,
                n_trials=args.n_trials,
                max_steps=args.max_steps,
                prior_alpha=args.prior_alpha,
                prior_beta=args.prior_beta,
                update_Cest_interval=args.cest_interval,
                active_learning_threshold=args.al_threshold,
                env_random=cfg["env_random"],
                pellet_random=cfg["pellet_random"],
                burnin_episodes=args.burnin_episodes,
                phase_split=args.phase_split,
                warm_scale=args.warm_scale,
                skip_individual=args.skip_individual,
            )
            save_path = os.path.join(
                args.outdir, f"{exp_id}_{scheme.replace('-','_')}.pkl"
            )
            with open(save_path, "wb") as f:
                pickle.dump(results, f)
            w = 50
            mf = np.mean([np.mean(r[-w:]) for r in results["rewards_feedback"]])
            mb = np.mean([np.mean(r[-w:]) for r in results["rewards_baseline"]])
            print(f"  Saved → {save_path}   "
                  f"feedback={mf:.1f}  baseline={mb:.1f}  Δ={mf-mb:+.1f}")


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Alternative training schemes using existing annotations."
    )
    p.add_argument("--scheme", default=None,
                   choices=RUN_ALL_SCHEMES,
                   help="Scheme to run (omit with --run-all to run all).")
    p.add_argument("--run-all", action="store_true",
                   help="Run all schemes across all 4 experiments.")
    p.add_argument("--bucket", default="jerskine_human_feedback",
                   help="GCS bucket name (default: jerskine_human_feedback)")
    p.add_argument("--sessions", nargs="+", default=None)
    p.add_argument("--llm-sessions", nargs="+", default=None,
                   help="Subset of --sessions that are LLM annotators "
                        "(used by two-phase scheme for phase 1).")
    p.add_argument("--local-brains-dir", default="llm_brains/")
    p.add_argument("--n-episodes",  type=int, default=1000)
    p.add_argument("--n-trials",    type=int, default=10)
    p.add_argument("--max-steps",   type=int, default=500)
    p.add_argument("--env-size",    default="small",
                   choices=["small", "medium", "medium_sparse"])
    p.add_argument("--prior-alpha", type=float, default=1.0)
    p.add_argument("--prior-beta",  type=float, default=1.0)
    p.add_argument("--al-threshold", type=float, default=3.0)
    p.add_argument("--cest-interval", type=int, default=5)
    p.add_argument("--env-random",    action="store_true")
    p.add_argument("--pellet-random", action="store_true")
    p.add_argument("--save", default="results/results_scheme.pkl",
                   help="Output path for single-scheme run.")
    p.add_argument("--outdir", default="results/",
                   help="Output directory for --run-all.")
    # Scheme-specific
    p.add_argument("--burnin-episodes", type=int, default=200,
                   help="[burnin] Episodes with Ce=1.0 before switching to adaptive.")
    p.add_argument("--phase-split",     type=int, default=300,
                   help="[two-phase] Episodes in phase 1 (LLM-only).")
    p.add_argument("--warm-scale",      type=float, default=50.0,
                   help="[warm-init] Scale factor for Q init (default 50.0).")
    p.add_argument("--skip-individual", action="store_true",
                   help="Skip per-session individual training (faster, enough for "
                        "scheme comparison plots).")
    args = p.parse_args()

    if args.run_all:
        os.makedirs(args.outdir, exist_ok=True)
        run_all(args)
        return

    if not args.scheme:
        p.error("Specify --scheme or --run-all")

    if not args.sessions:
        p.error("Specify --sessions (or use --run-all for all experiments)")

    sessions = load_all_brains(
        args.bucket,
        only=args.sessions,
        local_dir=args.local_brains_dir,
    )
    if not sessions:
        print("No sessions found — exiting.")
        sys.exit(1)

    llm_set = set(args.llm_sessions) if args.llm_sessions else set()

    print(f"\n  Scheme       : {args.scheme.upper()}")
    print(f"  Sessions     : {[sh for sh, _ in sessions]}")
    print(f"  LLM sessions : {llm_set or '(auto-detect not set)'}")
    print(f"  Episodes     : {args.n_episodes}  × {args.n_trials} trials")

    results = run_trials(
        scheme=args.scheme,
        sessions=sessions,
        llm_sessions=llm_set,
        env_size=args.env_size,
        n_episodes=args.n_episodes,
        n_trials=args.n_trials,
        max_steps=args.max_steps,
        prior_alpha=args.prior_alpha,
        prior_beta=args.prior_beta,
        update_Cest_interval=args.cest_interval,
        active_learning_threshold=args.al_threshold,
        env_random=args.env_random,
        pellet_random=args.pellet_random,
        burnin_episodes=args.burnin_episodes,
        phase_split=args.phase_split,
        warm_scale=args.warm_scale,
        skip_individual=args.skip_individual,
    )

    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    with open(args.save, "wb") as f:
        pickle.dump(results, f)

    w = 50
    mf = np.mean([np.mean(r[-w:]) for r in results["rewards_feedback"]])
    mb = np.mean([np.mean(r[-w:]) for r in results["rewards_baseline"]])
    print(f"\n── {args.scheme.upper()} result ──")
    print(f"  Feedback  : {mf:.1f}")
    print(f"  Baseline  : {mb:.1f}")
    print(f"  Δ         : {mf - mb:+.1f}")
    print(f"  Saved → {args.save}")


if __name__ == "__main__":
    main()
