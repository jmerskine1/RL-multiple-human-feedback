#!/usr/bin/env python3
"""
watch_bundles.py — Run locally during a study session.

Polls GCS every POLL_INTERVAL seconds. When it finds a session that needs
a new bundle (either exhausted or brand new), it immediately generates and
uploads one so participants aren't kept waiting.

Usage
-----
  python watch_bundles.py --bucket <your-gcs-bucket> [--interval 10]

Keep this running on your local machine throughout the study.
"""

import argparse
import time

from bundle_generator import generate_bundle, parse_participants_config
from gcs_utils import (
    bundle_exists, list_pending_sessions, list_brain_sessions, clear_pending,
)


def get_sessions_needing_bundles(bucket: str) -> set:
    """Sessions with a pending marker OR a brain but no bundle."""
    pending  = list_pending_sessions(bucket)
    with_brain = list_brain_sessions(bucket)
    all_sessions = pending | with_brain
    return {s for s in all_sessions if not bundle_exists(s, bucket)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bucket",   required=True)
    p.add_argument("--interval", type=int, default=10, help="Poll interval in seconds")
    p.add_argument("--n-feedbacks", type=int, default=10)
    p.add_argument("--mode",     default="count", choices=["count", "entropy"],
                   help="Default active-learning mode (overridden per-session by --participants-file)")
    p.add_argument("--env-size", default="small", choices=["small", "medium", "medium_sparse"])
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--env-random",    action="store_true",
                   help="Default: randomise start positions (overridden per-session by --participants-file)")
    p.add_argument("--pellet-random", action="store_true",
                   help="Default: randomise pellets (overridden per-session by --participants-file)")
    p.add_argument("--participants-file", default=None, metavar="FILE",
                   help="Config file with per-session mode/env/pellet settings "
                        "(e.g. participants.txt)")
    args = p.parse_args()

    per_session_cfg = {}
    if args.participants_file:
        per_session_cfg = parse_participants_config(args.participants_file)
        print(f"[watcher] loaded per-session config for "
              f"{len(per_session_cfg)} participant(s) from {args.participants_file}")

    print(f"[watcher] watching gs://{args.bucket} every {args.interval}s — Ctrl+C to stop\n")

    while True:
        try:
            sessions = get_sessions_needing_bundles(args.bucket)
            for session in sessions:
                cfg          = per_session_cfg.get(session, {})
                mode         = cfg.get("mode",          args.mode)
                env_random   = cfg.get("env_random",    args.env_random)
                pellet_random = cfg.get("pellet_random", args.pellet_random)

                print(f"[watcher] bundle needed for {session} "
                      f"(mode={mode}, env_random={env_random}, "
                      f"pellet_random={pellet_random}) — generating…")
                result = generate_bundle(
                    session_hash=session,
                    bucket_name=args.bucket,
                    n_feedbacks=args.n_feedbacks,
                    mode=mode,
                    env_size=args.env_size,
                    max_steps=args.max_steps,
                    force=True,
                    env_random=env_random,
                    pellet_random=pellet_random,
                )
                if result is not None:
                    clear_pending(session, args.bucket)
                    print(f"[watcher] ✓ bundle ready for {session}\n")
                else:
                    print(f"[watcher] ✗ could not generate bundle for {session} (all states labelled?)\n")
        except Exception as e:
            print(f"[watcher] error: {e}")

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
