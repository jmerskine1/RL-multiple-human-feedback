#!/usr/bin/env python3
"""
bundle_generator.py — LOCAL script for Option-B hybrid deployment.

Run this on your own machine (where matplotlib is available) to:
  1. Pull the latest agent brain from GCS.
  2. Run a Pacman episode and render every frame with matplotlib.
  3. Score each state with the active-learning utility function.
  4. Select the most informative states.
  5. Package everything into a bundle and upload to GCS.

The GCP Flask app (main_gcp.py) then serves the pre-rendered images and
applies lightweight numpy feedback updates — no matplotlib needed on GCP.

Usage
-----
  python bundle_generator.py \\
      --bucket  my-gcs-bucket \\
      --session participant_01 \\
      [--n-feedbacks 10] \\
      [--mode count|entropy] \\
      [--env-size small|medium] \\
      [--max-steps 500] \\
      [--all-sessions]   # generate for every session that has a brain in GCS

Authentication
--------------
  gcloud auth application-default login   # or set GOOGLE_APPLICATION_CREDENTIALS
"""

import argparse
import base64
import pickle
import json
from io import BytesIO
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

from trainer import PacmanTrainer
from feedback import get_active_utility
from gcs_utils import (
    download_brain, upload_bundle, get_labelled_states,
    bundle_exists, get_bucket
)


# ── per-participant config ────────────────────────────────────────────────────

def parse_participants_config(filepath: str) -> dict:
    """
    Parse a participants config file and return a dict of per-session settings.

    File format (all inline options are optional):
        SESSION_CODE  [mode=count|entropy]  [env=random]  [pellets=random]

    Returns:
        {session_code: {"mode": str, "env_random": bool, "pellet_random": bool}}

    Lines starting with # (after stripping) are ignored.
    """
    config = {}
    with open(filepath) as fh:
        for line in fh:
            line = line.split("#")[0].strip()
            if not line:
                continue
            parts = line.split()
            session = parts[0]
            opts = {"mode": "count", "env_random": False, "pellet_random": False,
                    "llm": None}
            for part in parts[1:]:
                if part == "mode=entropy":
                    opts["mode"] = "entropy"
                elif part == "mode=count":
                    opts["mode"] = "count"
                elif part == "env=random":
                    opts["env_random"] = True
                elif part == "pellets=random":
                    opts["pellet_random"] = True
                elif part.startswith("llm="):
                    opts["llm"] = part[4:]   # e.g. llm=claude-small → "claude-small"
            config[session] = opts
    return config


# ── helpers ───────────────────────────────────────────────────────────────────

def _encode_figure(fig) -> str:
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#000000")
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    buf.flush()
    plt.close(fig)
    return data


# ── core bundle generation ────────────────────────────────────────────────────

def generate_bundle(
    session_hash: str,
    bucket_name: str,
    n_feedbacks: int = 10,
    mode: str = "count",
    env_size: str = "small",
    max_steps: int = 500,
    max_attempts: int = 5,
    force: bool = False,
    env_random: bool = True,
    pellet_random: bool = False,
) -> dict | None:
    """
    Generate and upload a feedback bundle for one session.

    env_random    : randomise pacman + ghost start positions each episode.
    pellet_random : randomise which pellets are active each episode.

    Returns the bundle dict, or None if no unlabelled states were found.
    """
    if not force and bundle_exists(session_hash, bucket_name):
        print(f"[{session_hash}] bundle already exists in GCS (use --force to overwrite)")
        return None

    # ── 1. Load trainer + brain ───────────────────────────────────────────────
    trainer = PacmanTrainer(
        algID="tabQL_Cest_vi_t2",
        env_size=env_size,
        active_feedback_type=mode,
    )
    brain = download_brain(session_hash, bucket_name)
    if brain:
        trainer.load_brain(brain)
        print(f"[{session_hash}] brain loaded from GCS")
    else:
        print(f"[{session_hash}] no brain found — starting fresh")

    # ── 2. Load already-labelled states ──────────────────────────────────────
    labelled = get_labelled_states(session_hash, bucket_name)
    print(f"[{session_hash}] {len(labelled)} states already labelled")

    # ── 3. Run episode(s) until we find unlabelled states ────────────────────
    all_plots, all_obs, all_actions, all_status, all_valid_moves = [], [], [], [], []
    valid_indices = []

    for attempt in range(max_attempts):
        trainer.reset_episode(random=env_random, pellet_random=pellet_random)
        plots, obs_list, act_list, status_list, vm_list, fp = [], [], [], [], [], []

        print(f"[{session_hash}] attempt {attempt + 1}: running episode …", end=" ", flush=True)
        for _ in range(max_steps):
            obs = trainer.ob
            pac_dir = trainer.env.pacman.last_dir or 'e'
            pos = [list(trainer.env.pacman.pos), list(trainer.env.ghost.pos), pac_dir]
            fp.append(pos)

            # valid moves: [n, s, e, w]
            vm = [
                trainer.env.pacman.newPos(trainer.env.pacman.pos, d)
                != trainer.env.pacman.pos
                for d in ["n", "s", "e", "w"]
            ]
            vm_list.append(vm)

            trail      = [[p[0], p[1]] for p in fp[-3:-1]] if len(fp) >= 3 else None
            trail_dirs = [p[2]         for p in fp[-3:-1]] if len(fp) >= 3 else None
            fig, _ = trainer.env.plot(trail=trail, pacman_dir=pac_dir, trail_dirs=trail_dirs)
            plots.append(_encode_figure(fig))
            obs_list.append(obs)

            action_idx, _, _, done = trainer.step(feedback=[[]], update_Cest=False)
            act_list.append(action_idx)
            status_list.append(done)
            if done:
                break

        print(f"{len(plots)} frames")

        # Filter: not done, not previously labelled, no obs duplicates in this traj
        seen = set()
        cands = []
        for i, done in enumerate(status_list):
            if not done and obs_list[i] not in labelled and obs_list[i] not in seen:
                cands.append(i)
                seen.add(obs_list[i])

        if cands:
            all_plots, all_obs, all_actions = plots, obs_list, act_list
            all_status, all_valid_moves = status_list, vm_list
            valid_indices = cands
            print(f"[{session_hash}] {len(cands)} unlabelled candidate states found")
            break

    if not valid_indices:
        print(f"[{session_hash}] WARNING: no unlabelled states after {max_attempts} attempts")
        return None

    # ── 4. Active-learning state selection ───────────────────────────────────
    U = np.array([
        get_active_utility(all_obs[i], all_actions[i], trainer.agent, mode=mode)
        for i in valid_indices
    ])
    if np.std(U) > 0:
        U = (U - np.mean(U)) / np.std(U)

    N = min(len(valid_indices), n_feedbacks)
    if N == len(valid_indices):
        chosen = np.arange(len(U))
    else:
        # Top-N with small noise for variety
        chosen = np.argpartition(-U + np.random.randn(len(U)) * 0.1, N)[:N]

    selected_indices = sorted(valid_indices[i] for i in chosen)
    print(f"[{session_hash}] {len(selected_indices)} states selected for feedback")

    # ── 5. Package and upload ─────────────────────────────────────────────────
    bundle = {
        "session_hash":     session_hash,
        "selected_indices": selected_indices,
        "all_plots":        all_plots,
        "all_obs":          all_obs,
        "all_actions":      all_actions,
        "all_status":       all_status,
        "all_valid_moves":  all_valid_moves,
        "generated_at":     datetime.utcnow().isoformat(),
        "mode":             mode,
        "env_size":         env_size,
        "n_feedbacks":      n_feedbacks,
    }
    upload_bundle(session_hash, bundle, bucket_name)
    print(f"[{session_hash}] ✓ bundle uploaded ({len(selected_indices)} states, "
          f"{len(all_plots)} total frames)")
    return bundle


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Generate GCS feedback bundles locally.")
    p.add_argument("--bucket",      required=True,  help="GCS bucket name")
    p.add_argument("--session",     default=None,   help="Session hash (participant ID). "
                                                         "Omit with --all-sessions or --participants-file.")
    p.add_argument("--all-sessions",action="store_true",
                   help="Generate for every session that has a brain in GCS "
                        "(uses global --mode/--env-random/--pellet-random for all).")
    p.add_argument("--participants-file", default=None, metavar="FILE",
                   help="Participants config file (e.g. participants.txt). "
                        "Each line: SESSION_CODE [mode=count|entropy] [env=random] [pellets=random]. "
                        "Per-session settings override the global --mode/--env-random/--pellet-random flags.")
    p.add_argument("--n-feedbacks", type=int,  default=10, help="States per bundle")
    p.add_argument("--mode",        default="count",
                   choices=["count", "entropy"], help="Default active learning utility")
    p.add_argument("--env-size",    default="small",
                   choices=["small", "medium", "medium_sparse"])
    p.add_argument("--max-steps",   type=int,  default=500)
    p.add_argument("--env-random",  action="store_true",
                   help="Randomise pacman + ghost start positions (global default; "
                        "can be overridden per-session via --participants-file)")
    p.add_argument("--pellet-random", action="store_true",
                   help="Randomise active pellets (global default; "
                        "can be overridden per-session via --participants-file)")
    p.add_argument("--force",       action="store_true",
                   help="Overwrite existing bundles")
    args = p.parse_args()

    # Build per-session config: start from participants file (if given),
    # then fall back to global CLI flags for any session not listed there.
    per_session_cfg = {}
    if args.participants_file:
        per_session_cfg = parse_participants_config(args.participants_file)
        sessions = [s for s in per_session_cfg
                    if not s.startswith("LLM_") and not per_session_cfg[s].get("llm")]
        print(f"Loaded {len(sessions)} session(s) from {args.participants_file}")
    elif args.all_sessions:
        bucket = get_bucket(args.bucket)
        sessions = [
            b.name.split("/")[1]
            for b in bucket.list_blobs(prefix="brains/")
            if b.name.endswith("brain.pkl")
        ]
        if not sessions:
            print("No brains found in GCS — nothing to generate.")
            return
        print(f"Found {len(sessions)} session(s) in GCS: {sessions}")
    else:
        if not args.session:
            p.error("Provide --session <hash>, --all-sessions, or --participants-file")
        sessions = [args.session]

    for sh in sessions:
        cfg = per_session_cfg.get(sh, {})
        mode         = cfg.get("mode",         args.mode)
        env_random   = cfg.get("env_random",   args.env_random)
        pellet_random = cfg.get("pellet_random", args.pellet_random)
        print(f"[{sh}] mode={mode}  env_random={env_random}  pellet_random={pellet_random}")
        generate_bundle(
            session_hash=sh,
            bucket_name=args.bucket,
            n_feedbacks=args.n_feedbacks,
            mode=mode,
            env_size=args.env_size,
            max_steps=args.max_steps,
            force=args.force,
            env_random=env_random,
            pellet_random=pellet_random,
        )


if __name__ == "__main__":
    main()
