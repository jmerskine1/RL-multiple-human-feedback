#!/usr/bin/env python3
"""
count_states.py — Read-only snapshot of labelled-state counts from GCS.

For every session in participants.txt, downloads its labelled_states.json
and prints how many unique states have been collected.  Nothing is written.

Usage
-----
  python count_states.py --bucket jerskine_human_feedback
  python count_states.py --bucket jerskine_human_feedback --participants-file participants.txt
  python count_states.py --bucket jerskine_human_feedback --max-states 100   # show % progress
  python count_states.py --bucket jerskine_human_feedback --human-only
  python count_states.py --bucket jerskine_human_feedback --llm-only
"""

import argparse
import json
import pickle
import sys

import numpy as np

from gcs_utils import get_bucket
from bundle_generator import parse_participants_config

# ── experiment grouping ───────────────────────────────────────────────────────

EXP_LABELS = {
    ("count",   False): "Exp 1 · count / fixed",
    ("entropy", False): "Exp 2 · entropy / fixed",
    ("count",   True):  "Exp 3 · count / random",
    ("entropy", True):  "Exp 4 · entropy / random",
}

# ── helpers ───────────────────────────────────────────────────────────────────

def fetch_count_human(session_hash: str, bucket_name: str) -> int:
    """Count from labelled_states.json — written by the web app on each human submit."""
    blob = get_bucket(bucket_name).blob(
        f"feedback/{session_hash}/labelled_states.json"
    )
    if not blob.exists():
        return 0
    return len(json.loads(blob.download_as_text()))


def fetch_count_llm(session_code: str, llm_name: str, bucket_name: str) -> int:
    """Count from the brain pkl — written by llm_annotate.py (never touches labelled_states.json).

    Tries session_code first, then the legacy LLM_<name> key.
    Falls back to hp-based count if n_annotated_states is absent (old brains).
    """
    bucket = get_bucket(bucket_name)
    brain = None
    for key in (session_code, f"LLM_{llm_name}"):
        blob = bucket.blob(f"brains/{key}/brain.pkl")
        if blob.exists():
            brain = pickle.loads(blob.download_as_bytes())
            break
    if brain is None:
        return 0
    n = brain.get("n_annotated_states")
    if n is not None:
        return int(n)
    # Legacy brain: estimate from hp matrix
    hp = brain.get("hp")
    if hp is not None:
        return int(np.any(hp[0] > 0, axis=1).sum())
    return 0


def fetch_count(session_code: str, opts: dict, bucket_name: str) -> int:
    """Dispatch to the right counter based on whether this is an LLM session."""
    llm = opts.get("llm")
    if llm:
        return fetch_count_llm(session_code, llm, bucket_name)
    return fetch_count_human(session_code, bucket_name)


def bar(n: int, total: int, width: int = 20) -> str:
    filled = int(width * n / total) if total else 0
    return f"[{'█' * filled}{'░' * (width - filled)}]"


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Count labelled states per session (read-only).")
    p.add_argument("--bucket", required=True, help="GCS bucket name")
    p.add_argument("--participants-file", default="participants.txt",
                   help="Participants config file (default: participants.txt)")
    p.add_argument("--max-states", type=int, default=0,
                   help="Target states per session — shows %% progress bar if set")
    p.add_argument("--human-only", action="store_true",
                   help="Only show human sessions (no llm= lines)")
    p.add_argument("--llm-only", action="store_true",
                   help="Only show LLM sessions (only llm= lines)")
    args = p.parse_args()

    if args.human_only and args.llm_only:
        p.error("--human-only and --llm-only are mutually exclusive.")

    cfg = parse_participants_config(args.participants_file)

    # Apply human/llm filter
    if args.human_only:
        cfg = {s: o for s, o in cfg.items() if not o.get("llm")}
    elif args.llm_only:
        cfg = {s: o for s, o in cfg.items() if o.get("llm")}

    if not cfg:
        print("No sessions found — check your participants file.")
        sys.exit(0)

    # Group by experiment
    groups: dict[str, list[tuple[str, dict]]] = {}
    ungrouped: list[tuple[str, dict]] = []
    for session, opts in cfg.items():
        mode = opts.get("mode", "count")
        is_random = bool(opts.get("env_random") or opts.get("pellet_random"))
        key = EXP_LABELS.get((mode, is_random))
        if key:
            groups.setdefault(key, []).append((session, opts))
        else:
            ungrouped.append((session, opts))

    max_s = args.max_states
    col_w = 22  # session code column width

    def print_session(session: str, opts: dict):
        n = fetch_count(session, opts, args.bucket)
        llm_tag = f"  [{opts['llm']}]" if opts.get("llm") else ""
        if max_s:
            pct = min(100, int(100 * n / max_s))
            progress = f"  {bar(n, max_s)} {n:>4}/{max_s}  ({pct:>3}%)"
        else:
            progress = f"  {n:>4} states"
        print(f"  {session:<{col_w}}{progress}{llm_tag}")

    grand_total = 0
    grand_sessions = 0

    # Print ungrouped first (e.g. original human codes with no mode)
    if ungrouped:
        print("\n── Ungrouped sessions ───────────────────────────────────────────")
        for session, opts in ungrouped:
            print_session(session, opts)
            grand_total += fetch_count(session, opts, args.bucket)
            grand_sessions += 1

    # Print grouped experiments
    ordered_keys = [
        "Exp 1 · count / fixed",
        "Exp 2 · entropy / fixed",
        "Exp 3 · count / random",
        "Exp 4 · entropy / random",
    ]
    for key in ordered_keys:
        sessions_in_group = groups.get(key)
        if not sessions_in_group:
            continue

        subtotal = 0
        rows = []
        for session, opts in sessions_in_group:
            n = fetch_count(session, opts, args.bucket)
            subtotal += n
            grand_total += n
            grand_sessions += 1
            llm_tag = f"  [{opts['llm']}]" if opts.get("llm") else ""
            if max_s:
                pct = min(100, int(100 * n / max_s))
                progress = f"  {bar(n, max_s)} {n:>4}/{max_s}  ({pct:>3}%)"
            else:
                progress = f"  {n:>4} states"
            rows.append(f"  {session:<{col_w}}{progress}{llm_tag}")

        print(f"\n── {key} ({'subtotal: ' + str(subtotal) + ' states'})")
        print("\n".join(rows))

    print()
    print("=" * 68)
    if max_s:
        target = max_s * grand_sessions
        pct = int(100 * grand_total / target) if target else 0
        print(f"  TOTAL  {grand_total:>6} / {target} states across {grand_sessions} sessions  ({pct}%)")
    else:
        print(f"  TOTAL  {grand_total:>6} states across {grand_sessions} sessions")
    print("=" * 68)


if __name__ == "__main__":
    main()
