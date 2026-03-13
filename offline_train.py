#!/usr/bin/env python3
"""
offline_train.py — Aggregate feedback from all participants and compare models.

Each participant's per-session brain (nTrainer=1) is loaded from GCS and their
hp/hm feedback arrays are stacked into a single combined agent with nTrainer=N.
The Variational Inference C-estimation assigns a consistency weight Ce[m] ∈ [0,1]
to each participant, downweighting noisy trainers automatically.

Two training modes are available (controlled by --training-mode):

  replay (default) — mirrors main_oracle.py exactly, but using pre-gathered
      human feedback instead of a live oracle. At each training step, if the
      current state was labelled during the study AND active learning says it
      still needs coverage, a Feedback object is injected into the step —
      identical to how the oracle injects feedback. hp/hm accumulate during
      training so Ce and psi_for_hr/hw strengthen over time. Uses a fixed
      start state and a proper terminal Q-update, exactly matching the oracle
      training loop.

  static — original behaviour: hp/hm are frozen at study values and used only
      as a fixed shaping prior via psi_for_hr/hw. Simpler but weaker because
      the shaping signal cannot grow during training.

Results are saved locally as a .pkl file (no GCS upload needed).

Pipeline
--------
1.  Load specified participant brains from gs://<bucket>/brains/
2.  Stack hp/hm:  [1, nS, nA] per participant → [N, nS, nA]
3.  Run VI C-estimation EM loop to get Ce[m] per participant
4.  Train WITH feedback for --n-episodes episodes, recording reward per episode
5.  Train WITHOUT feedback for --n-episodes episodes (fresh agent, same env)
6.  Print consistency report + performance comparison
7.  Save both Q-tables and learning curves locally

Usage
-----
  # Replay mode (recommended — matches oracle training setup):
  python offline_train.py \\
      --bucket   <your-gcs-bucket> \\
      --sessions P001 P002 P003 \\
      --n-episodes 1000

  # Static mode (original behaviour):
  python offline_train.py \\
      --bucket   <your-gcs-bucket> \\
      --sessions P001 P002 P003 \\
      --n-episodes 1000 \\
      --training-mode static

Authentication
--------------
  gcloud auth application-default login
"""

import os
import argparse
import pickle
import numpy as np
from scipy.special import psi as scipy_psi

from feedback import Feedback
from trainer import PacmanTrainer
from gcs_utils import list_brain_sessions, download_brain


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all_brains(bucket_name: str, only: list = None,
                    local_dir: str = None) -> list:
    """
    Load participant brains from GCS (and optionally a local directory).

    Args:
        bucket_name:  GCS bucket name
        only:         If provided, load only these session names
                      (e.g. ['P001', 'P002', 'LLM_claude-small']).
                      If None, load every brain found in GCS brains/.
        local_dir:    Optional local directory to fall back to when GCS lookup
                      fails.  Files must be named  {SESSION}_brain.pkl.
                      Useful for LLM brains saved locally by llm_annotate.py.

    Returns [(session_hash, brain_dict), ...]
    """
    if only:
        # Preserve case — LLM session names are mixed-case (e.g. LLM_claude-small)
        sessions = [s.strip() for s in only]
        print(f"  Filtering to {len(sessions)} specified session(s): {sessions}")
    else:
        sessions = sorted(list_brain_sessions(bucket_name))
        if not sessions:
            raise RuntimeError("No brains found in GCS — run the study first.")
        print(f"  Found {len(sessions)} session(s) in bucket")

    result = []
    for sh in sessions:
        brain = download_brain(sh, bucket_name)

        # Fallback: try local directory (e.g. for LLM brains saved by llm_annotate.py)
        if brain is None and local_dir:
            local_path = os.path.join(local_dir, f"{sh}_brain.pkl")
            if os.path.exists(local_path):
                with open(local_path, "rb") as f:
                    brain = pickle.load(f)
                print(f"  [{sh}] loaded from local file: {local_path}")

        if brain is None:
            print(f"  [{sh}] not found in GCS or local dir — skipping")
            continue

        hp = np.array(brain.get('hp', []))
        total_feedback = float(np.sum(hp)) if hp.size > 0 else 0.0
        if total_feedback == 0.0:
            print(f"  [{sh}] no feedback recorded — skipping")
            continue

        result.append((sh, brain))
        print(f"  [{sh}]  feedback mass: {total_feedback:.1f}")

    return result


# ── Agent construction ────────────────────────────────────────────────────────

def build_combined_agent(
    trainer: PacmanTrainer,
    sessions: list,
    q_init: str = "zeros",
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> None:
    """
    Inject a combined multi-trainer state into trainer.agent.

    - hp and hm are stacked from [1, nS, nA] per participant → [N, nS, nA]
    - Q is initialised from zero (cleanest for fair comparison) or average of
      participants' Q-tables
    - All VI sufficient statistics are initialised ready for C-estimation
    """
    agent = trainer.agent
    N = len(sessions)

    def _extract(brain, key):
        arr = np.array(brain[key])
        return arr[0] if arr.ndim == 3 else arr   # [1,nS,nA] → [nS,nA]

    all_hp = [_extract(b, 'hp') for _, b in sessions]
    all_hm = [_extract(b, 'hm') for _, b in sessions]
    combined_hp = np.stack(all_hp)   # [N, nS, nA]
    combined_hm = np.stack(all_hm)
    nS, nA = combined_hp.shape[1], combined_hp.shape[2]

    if q_init == "average":
        q_tables = [np.array(b['Q']) for _, b in sessions if b.get('Q') is not None]
        initial_Q = np.mean(q_tables, axis=0) if q_tables else np.zeros((nS, nA))
        print(f"  Q: average of {len(q_tables)} participant table(s)")
    else:
        initial_Q = np.zeros((nS, nA))
        print("  Q: zeros (fair baseline comparison)")

    all_nsa = [np.array(b.get('Nsa', np.zeros((nS, nA)))) for _, b in sessions]

    agent.Q        = initial_Q
    agent.hp       = combined_hp
    agent.hm       = combined_hm
    agent.Nsa      = np.sum(all_nsa, axis=0)
    agent.nTrainer = N
    agent.a        = np.ones(N) * prior_alpha
    agent.b        = np.ones(N) * prior_beta

    agent.Ce                    = np.ones(N) * 0.5
    agent.sum_of_right_feedback = np.zeros(N)
    agent.sum_of_wrong_feedback = np.zeros(N)
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


# ── C estimation ──────────────────────────────────────────────────────────────

def run_c_estimation(trainer: PacmanTrainer) -> np.ndarray:
    """Run the VI EM loop on existing hp/hm without adding new feedback."""
    N = trainer.agent.nTrainer
    trainer.agent.act(0, 0, 0.0, False, [[] for _ in range(N)], 0.5, update_Cest=True)
    return trainer.agent.Ce.copy()


# ── Training loops ────────────────────────────────────────────────────────────

def train_with_feedback(
    trainer: PacmanTrainer,
    n_episodes: int,
    max_steps: int = 500,
    update_Cest_interval: int = 5,
    adaptive_cest: bool = True,
) -> list:
    """
    Static mode: train using frozen hp/hm as a fixed shaping prior.
    C is re-estimated every update_Cest_interval episodes (unless
    adaptive_cest=False, in which case Ce is frozen at its initial value).
    Returns list of total reward per episode.
    """
    N = trainer.agent.nTrainer
    empty_fb = [[] for _ in range(N)]
    rewards = []

    for ep in range(n_episodes):
        trainer.reset_episode()
        steps = 0
        prev_action = 0
        while not trainer.done and steps < max_steps:
            _, _, _, done = trainer.step(feedback=empty_fb, update_Cest=False)
            steps += 1
            if done:
                break

        # Explicit terminal Q-update (fixes missing terminal reward bug),
        # then optionally re-estimate C
        update_Cest = adaptive_cest and ((ep + 1) % update_Cest_interval == 0)
        trainer.agent.act(0, trainer.ob, trainer.rw, trainer.done,
                          empty_fb, 0.5, update_Cest=update_Cest)
        trainer.agent.prev_obs = None

        rewards.append(float(trainer.totRW))

        if (ep + 1) % max(1, n_episodes // 10) == 0:
            ce_str = ", ".join(f"{c:.3f}" for c in trainer.agent.Ce)
            print(f"  [static]  ep {ep+1:>4}/{n_episodes}  "
                  f"reward={trainer.totRW:>7.2f}  steps={steps:<4}  Ce=[{ce_str}]")

    return rewards


def train_individual_condition(
    session_hash: str,
    brain: dict,
    env_size: str,
    n_episodes: int,
    max_steps: int = 500,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    training_mode: str = "replay",
    active_learning_threshold: float = 3.0,
    update_Cest_interval: int = 5,
    adaptive_cest: bool = True,
) -> tuple:
    """
    Train with a single participant's feedback in isolation.
    Reuses build_combined_agent with nTrainer=1.
    Returns (rewards_list, Ce_scalar).
    """
    trainer = PacmanTrainer(
        algID="tabQL_Cest_vi_t2",
        env_size=env_size,
        prior_alpha=prior_alpha,
        prior_beta=prior_beta,
    )
    build_combined_agent(
        trainer, [(session_hash, brain)],
        q_init="zeros",
        prior_alpha=prior_alpha,
        prior_beta=prior_beta,
    )
    run_c_estimation(trainer)
    ce = float(trainer.agent.Ce[0])

    if training_mode == "replay":
        # Freeze study data as lookup — hp/hm will grow during training
        hp0 = trainer.agent.hp.copy()
        hm0 = trainer.agent.hm.copy()
        rewards = train_with_human_replay(
            trainer, hp0, hm0,
            n_episodes=n_episodes,
            max_steps=max_steps,
            update_Cest_interval=update_Cest_interval,
            active_learning_threshold=active_learning_threshold,
            adaptive_cest=adaptive_cest,
        )
        # Print final line (train_with_human_replay already prints per-interval)
    else:
        # Static mode — frozen hp/hm, C-estimation each episode
        N = trainer.agent.nTrainer
        empty_fb = [[] for _ in range(N)]
        rewards = []

        for ep in range(n_episodes):
            trainer.reset_episode()
            steps = 0
            while not trainer.done and steps < max_steps:
                trainer.step(feedback=empty_fb, update_Cest=False)
                steps += 1

            # Terminal Q-update, then C-estimation
            update_Cest = adaptive_cest and ((ep + 1) % update_Cest_interval == 0)
            trainer.agent.act(0, trainer.ob, trainer.rw, trainer.done,
                              empty_fb, 0.5, update_Cest=update_Cest)
            trainer.agent.prev_obs = None
            rewards.append(float(trainer.totRW))

            if (ep + 1) % max(1, n_episodes // 10) == 0:
                print(f"  [{session_hash:<10}]  ep {ep+1:>4}/{n_episodes}  "
                      f"reward={trainer.totRW:>7.2f}  steps={steps:<4}  "
                      f"Ce={trainer.agent.Ce[0]:.3f}")

    return rewards, ce


def _human_feedback_for_state(
    hp0: np.ndarray,
    hm0: np.ndarray,
    state: int,
    n_trainers: int,
    nsa_row: np.ndarray,
    active_learning_threshold: float,
) -> list:
    """
    Build a per-trainer feedback list for `state` from frozen study hp0/hm0.

    Mirrors generate_feedback() in main_oracle.py:
      - Uses count-based active learning (Nsa < threshold) to decide whether
        to inject feedback for this state at all.
      - For each trainer, if the state has any human label, creates a Feedback
        object whose good/bad action sets are derived from the sign of hp-hm.
      - Returns [[] for _ in range(n_trainers)] when no feedback is appropriate.
    """
    # Count-based active learning: skip state if it has been visited enough
    if np.min(nsa_row) >= active_learning_threshold:
        return [[] for _ in range(n_trainers)]

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


def train_with_human_replay(
    trainer: PacmanTrainer,
    hp0: np.ndarray,
    hm0: np.ndarray,
    n_episodes: int,
    max_steps: int = 500,
    update_Cest_interval: int = 5,
    active_learning_threshold: float = 3.0,
    adaptive_cest: bool = True,
) -> list:
    """
    Train using human feedback replayed at each step — mirrors main_oracle.py.

    Instead of calling an oracle, we look up the pre-gathered human labels in
    the frozen arrays hp0/hm0. Whenever the agent visits a state that was
    labelled during the study AND active learning says it still needs coverage
    (Nsa < threshold), a Feedback object is injected into that step.

    hp/hm accumulate during training (just as in the oracle run), so Ce and the
    policy-shaping psi values grow stronger as the agent revisits labelled
    states. C-estimation updates every `update_Cest_interval` episodes, unless
    adaptive_cest=False in which case Ce is frozen at its initial value and all
    participants contribute with equal, fixed weights throughout training.

    Uses a fixed start state (random=False) and applies the terminal Q-update
    explicitly — both matching the main_oracle.py training loop exactly.
    """
    N = trainer.agent.nTrainer
    rewards = []

    for ep in range(n_episodes):
        trainer.reset_episode(random=False)   # fixed start — matches oracle
        prev_action = 0
        steps = 0

        while not trainer.done and steps < max_steps:
            state = trainer.ob

            # Generate feedback from the frozen human data for this state
            fb = _human_feedback_for_state(
                hp0, hm0, state, N,
                trainer.agent.Nsa[state, :],
                active_learning_threshold,
            )

            action_idx, ob, rw, done = trainer.step(feedback=fb,
                                                     update_Cest=False)
            prev_action = action_idx
            steps += 1

            if done or steps >= max_steps:
                # Explicit terminal Q-update — matches main_oracle.py line:
                #   trainer.agent.act(action_idx, ob, rw, done, fb, C, ...)
                update_Cest = adaptive_cest and ((ep + 1) % update_Cest_interval == 0)
                trainer.agent.act(action_idx, ob, rw, done,
                                  fb, 0.5, update_Cest=update_Cest)
                trainer.agent.prev_obs = None
                break

        rewards.append(float(trainer.totRW))

        if (ep + 1) % max(1, n_episodes // 10) == 0:
            ce_str = ", ".join(f"{c:.3f}" for c in trainer.agent.Ce)
            print(f"  [replay]  ep {ep+1:>4}/{n_episodes}  "
                  f"reward={trainer.totRW:>7.2f}  steps={steps:<4}  "
                  f"Ce=[{ce_str}]")

    return rewards


def train_without_feedback(
    env_size: str,
    n_episodes: int,
    max_steps: int = 500,
) -> list:
    """
    Train a fresh agent for n_episodes with pure Q-learning (no human feedback).
    Returns list of total reward per episode.
    """
    baseline = PacmanTrainer(
        algID="tabQL_Cest_vi_t2",
        env_size=env_size,
    )
    # Initialise Q by running one dummy step (triggers the Q is None init block)
    baseline.agent.act(0, 0, 0.0, False, [[]], 0.5, update_Cest=False)
    rewards = []

    for ep in range(n_episodes):
        baseline.reset_episode()
        steps = 0
        while not baseline.done and steps < max_steps:
            baseline.step(feedback=[[]], update_Cest=False)
            steps += 1

        rewards.append(float(baseline.totRW))

        if (ep + 1) % max(1, n_episodes // 10) == 0:
            print(f"  [baseline]   ep {ep+1:>4}/{n_episodes}  "
                  f"reward={baseline.totRW:>7.2f}  steps={steps:<4}")

    return rewards, baseline


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_consistency_report(sessions: list, Ce: np.ndarray) -> None:
    print("\n" + "=" * 58)
    print(f"  {'Rank':<5} {'Session':<22} {'Ce':>6}   Signal")
    print("-" * 58)
    for rank, m in enumerate(np.argsort(-Ce)):
        sh  = sessions[m][0]
        ce  = Ce[m]
        tag = ("✓ reliable" if ce >= 0.75 else
               "~ moderate" if ce >= 0.55 else
               "✗ noisy")
        print(f"  {rank+1:<5} {sh:<22} {ce:>6.3f}   {tag}")
    print("=" * 58)
    print(f"  Mean Ce: {Ce.mean():.3f}   Std: {Ce.std():.3f}\n")


def print_performance_report(
    rewards_fb: list,
    rewards_bl: list,
    window: int = 10,
) -> None:
    """Compare mean reward over the last `window` episodes for both conditions."""
    def _tail_mean(r):
        return float(np.mean(r[-window:])) if len(r) >= window else float(np.mean(r))

    fb_mean = _tail_mean(rewards_fb)
    bl_mean = _tail_mean(rewards_bl)
    delta   = fb_mean - bl_mean
    pct     = (delta / abs(bl_mean) * 100) if bl_mean != 0 else float('nan')

    print("── Performance comparison (mean reward, last "
          f"{min(window, len(rewards_fb))} episodes) ──")
    print(f"  With    human feedback:  {fb_mean:>8.2f}")
    print(f"  Without human feedback:  {bl_mean:>8.2f}")
    print(f"  Δ (feedback − baseline): {delta:>+8.2f}  ({pct:+.1f}%)")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Aggregate participant feedback and compare with/without training."
    )
    p.add_argument("--bucket",       required=True,
                   help="GCS bucket name")
    p.add_argument("--sessions",     nargs="+", default=None, metavar="SESSION",
                   help="Participant codes to include (e.g. P001 P002 P003). "
                        "Omit to use all brains in the bucket.")
    p.add_argument("--sessions-file", default=None, metavar="FILE",
                   help="Text file with one session code per line (comments with # "
                        "are ignored). Merged with any --sessions values.")
    p.add_argument("--n-episodes",   type=int, default=100,
                   help="Training episodes per condition (default: 100)")
    p.add_argument("--max-steps",    type=int, default=500,
                   help="Max steps per episode (default: 500)")
    p.add_argument("--q-init",       default="zeros",
                   choices=["zeros", "average"],
                   help="Q init for feedback condition: 'zeros' (default, fairest "
                        "comparison) or 'average' of participant tables")
    p.add_argument("--env-size",     default="small",
                   choices=["small", "medium", "medium_sparse"])
    p.add_argument("--prior-alpha",  type=float, default=1.0,
                   help="Beta prior α for C estimation (default 1.0 = flat)")
    p.add_argument("--prior-beta",   type=float, default=1.0,
                   help="Beta prior β for C estimation (default 1.0 = flat)")
    p.add_argument("--local-brains-dir", default=None, metavar="DIR",
                   help="Local directory to search for brain files as a fallback "
                        "when GCS lookup fails (e.g. --local-brains-dir llm_brains/). "
                        "Files must be named {SESSION}_brain.pkl.  "
                        "Useful for LLM brains saved locally by llm_annotate.py.")
    p.add_argument("--save",         default="results/results.pkl",
                   help="Local path to save results (default: results/results.pkl)")
    p.add_argument("--report-window", type=int, default=10,
                   help="Episodes to average for final performance report (default: 10)")
    p.add_argument("--training-mode", default="replay",
                   choices=["replay", "static"],
                   help="replay (default): inject human labels at each step, matching "
                        "main_oracle.py — hp/hm grow, Ce strengthens during training. "
                        "static: frozen hp/hm used only as a fixed shaping prior.")
    p.add_argument("--al-threshold",  type=float, default=3.0,
                   help="Active-learning visit threshold for replay mode: "
                        "feedback injected only while Nsa < threshold (default: 3.0)")
    p.add_argument("--cest-interval", type=int, default=5,
                   help="Re-estimate C every N episodes (default: 5, matches oracle)")
    p.add_argument("--no-cest", action="store_true",
                   help="Disable adaptive Ce estimation during training. Ce is "
                        "initialised once from the study data then held fixed, so all "
                        "participants contribute with equal, unchanging weights. "
                        "Useful as an ablation to isolate the benefit of Ce weighting.")
    p.add_argument("--n-trials", type=int, default=1,
                   help="Number of independent training runs with different random "
                        "seeds (default: 1). Use ≥10 to get reliable variance estimates. "
                        "Brains are loaded once; the training loop is repeated per trial.")
    args = p.parse_args()

    # Merge --sessions-file into args.sessions
    if args.sessions_file:
        file_sessions = []
        with open(args.sessions_file) as fh:
            for line in fh:
                code = line.split("#")[0].strip()
                if code:
                    file_sessions.append(code)
        args.sessions = list(dict.fromkeys((args.sessions or []) + file_sessions))

    adaptive_cest = not args.no_cest
    cest_label = "fixed (--no-cest)" if args.no_cest else f"adaptive (every {args.cest_interval} eps)"
    print(f"\n  Training mode : {args.training_mode.upper()}")
    print(f"  Ce estimation : {cest_label}")

    # ── 1. Load participant brains ─────────────────────────────────────────────
    print(f"\n── Loading participant brains from gs://{args.bucket}/brains/ ──")
    sessions = load_all_brains(args.bucket, only=args.sessions,
                               local_dir=args.local_brains_dir)
    if not sessions:
        print("No usable sessions found. Exiting.")
        return
    print(f"\n  {len(sessions)} participant(s) with feedback\n")

    # ── Trials loop (repeats steps 2–6 with different random seeds) ───────────
    all_rewards_fb         = []
    all_rewards_bl         = []
    all_rewards_individual = {sh: [] for sh, _ in sessions}
    all_Ce                 = []
    all_ce_individual      = {sh: [] for sh, _ in sessions}
    last_fb_trainer = last_bl_trainer = None

    for trial in range(args.n_trials):
        if args.n_trials > 1:
            print(f"\n{'═'*60}")
            print(f"  Trial {trial + 1} / {args.n_trials}")
            print(f"{'═'*60}")
        np.random.seed(trial)

        # ── 2. Build combined feedback agent ──────────────────────────────────
        if args.n_trials == 1:
            print("── Building combined multi-trainer agent ──")
        fb_trainer = PacmanTrainer(
            algID="tabQL_Cest_vi_t2",
            env_size=args.env_size,
            prior_alpha=args.prior_alpha,
            prior_beta=args.prior_beta,
        )
        build_combined_agent(
            fb_trainer, sessions,
            q_init=args.q_init,
            prior_alpha=args.prior_alpha,
            prior_beta=args.prior_beta,
        )

        # ── 3. C estimation ───────────────────────────────────────────────────
        if adaptive_cest:
            if args.n_trials == 1:
                print("\n── Running C-estimation ──")
            Ce = run_c_estimation(fb_trainer)
            if args.n_trials == 1:
                print_consistency_report(sessions, Ce)
        else:
            # --no-cest: skip VI estimation entirely — assume all trainers are
            # fully reliable (Ce = 1.0).  This is the correct ablation: removing
            # the Ce weighting effect, not just freezing a biased initial estimate.
            N = len(sessions)
            Ce = np.ones(N)
            fb_trainer.agent.Ce = Ce.copy()
            if args.n_trials == 1:
                print("\n── C-estimation skipped (--no-cest) — all Ce set to 1.0 ──")
        all_Ce.append(Ce)

        # ── 4. Train WITH feedback (combined, all participants) ───────────────
        if args.training_mode == "replay":
            if args.n_trials == 1:
                print(f"── Training WITH feedback — combined, replay mode "
                      f"({args.n_episodes} episodes) ──")
            hp0 = fb_trainer.agent.hp.copy()
            hm0 = fb_trainer.agent.hm.copy()
            rewards_fb = train_with_human_replay(
                fb_trainer, hp0, hm0,
                n_episodes=args.n_episodes,
                max_steps=args.max_steps,
                update_Cest_interval=args.cest_interval,
                active_learning_threshold=args.al_threshold,
                adaptive_cest=adaptive_cest,
            )
        else:
            if args.n_trials == 1:
                print(f"── Training WITH feedback — combined, static mode "
                      f"({args.n_episodes} episodes) ──")
            rewards_fb = train_with_feedback(
                fb_trainer, args.n_episodes,
                max_steps=args.max_steps,
                update_Cest_interval=args.cest_interval,
                adaptive_cest=adaptive_cest,
            )
        all_rewards_fb.append(rewards_fb)
        last_fb_trainer = fb_trainer

        # ── 5. Train WITHOUT feedback (baseline) ──────────────────────────────
        if args.n_trials == 1:
            print(f"\n── Training WITHOUT feedback — baseline ({args.n_episodes} episodes) ──")
        rewards_bl, bl_trainer = train_without_feedback(
            args.env_size, args.n_episodes, max_steps=args.max_steps
        )
        all_rewards_bl.append(rewards_bl)
        last_bl_trainer = bl_trainer

        # ── 6. Train each participant individually ────────────────────────────
        for sh, brain in sessions:
            if args.n_trials == 1:
                print(f"\n── Training — individual: {sh} ({args.n_episodes} episodes) ──")
            rw, ce = train_individual_condition(
                sh, brain,
                env_size=args.env_size,
                n_episodes=args.n_episodes,
                max_steps=args.max_steps,
                prior_alpha=args.prior_alpha,
                prior_beta=args.prior_beta,
                training_mode=args.training_mode,
                active_learning_threshold=args.al_threshold,
                update_Cest_interval=args.cest_interval,
                adaptive_cest=adaptive_cest,
            )
            all_rewards_individual[sh].append(rw)
            all_ce_individual[sh].append(ce)

        if args.n_trials > 1:
            mean_fb = np.mean(all_rewards_fb[-1][-args.report_window:])
            mean_bl = np.mean(all_rewards_bl[-1][-args.report_window:])
            print(f"  Trial {trial+1} done — "
                  f"feedback={mean_fb:.3f}  baseline={mean_bl:.3f}")

    # ── 7. Performance report ─────────────────────────────────────────────────
    print()
    mean_rewards_fb = np.mean(all_rewards_fb, axis=0)
    mean_rewards_bl = np.mean(all_rewards_bl, axis=0)
    print_performance_report(mean_rewards_fb, mean_rewards_bl, window=args.report_window)
    if args.n_trials > 1:
        std_fb = np.std([np.mean(r[-args.report_window:]) for r in all_rewards_fb])
        std_bl = np.std([np.mean(r[-args.report_window:]) for r in all_rewards_bl])
        print(f"  Across {args.n_trials} trials — "
              f"feedback std={std_fb:.3f}  baseline std={std_bl:.3f}")

    # ── 8. Save results locally ────────────────────────────────────────────────
    results = {
        "sessions":            [sh for sh, _ in sessions],
        "Ce":                  np.mean(all_Ce, axis=0),
        "Ce_individual":       {sh: float(np.mean(all_ce_individual[sh]))
                                for sh in all_ce_individual},
        "rewards_feedback":    all_rewards_fb,    # list of arrays (one per trial)
        "rewards_baseline":    all_rewards_bl,    # list of arrays (one per trial)
        "rewards_individual":  {sh: all_rewards_individual[sh]
                                for sh, _ in sessions},
        "Q_feedback":          last_fb_trainer.agent.Q,
        "Q_baseline":          last_bl_trainer.agent.Q,
        "hp":                  last_fb_trainer.agent.hp,
        "hm":                  last_fb_trainer.agent.hm,
        "n_episodes":          args.n_episodes,
        "n_trials":            args.n_trials,
        "env_size":            args.env_size,
        "q_init":              args.q_init,
    }
    with open(args.save, "wb") as f:
        pickle.dump(results, f)
    print(f"── Results saved → {args.save} ──")
    print(f"  Load with:  import pickle; r = pickle.load(open('{args.save}','rb'))")
    print("  Keys: sessions, Ce, rewards_feedback, rewards_baseline, "
          "Q_feedback, Q_baseline\n")


if __name__ == "__main__":
    main()
