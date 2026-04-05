#!/usr/bin/env python3
"""
manage_codes.py — Add/list/remove/reset participant access codes in GCS.

Usage
-----
  python manage_codes.py --bucket <your-gcs-bucket> add P001 P002 P003
  python manage_codes.py --bucket <your-gcs-bucket> add --participants-file participants.txt
  python manage_codes.py --bucket <your-gcs-bucket> list
  python manage_codes.py --bucket <your-gcs-bucket> remove P001
  python manage_codes.py --bucket <your-gcs-bucket> reset P001
  python manage_codes.py --bucket <your-gcs-bucket> reset --participants-file participants.txt
  python manage_codes.py --bucket <your-gcs-bucket> status
"""

import argparse
import json
from gcs_utils import get_valid_codes, add_valid_code, get_bucket


def _codes_from_participants_file(filepath: str) -> list[str]:
    """Return all session codes from a participants.txt (skips comments, LLM sessions)."""
    from bundle_generator import parse_participants_config
    cfg = parse_participants_config(filepath)
    return [s for s, opts in cfg.items() if not opts.get("llm")]


def remove_code(code: str, bucket_name: str) -> None:
    codes = get_valid_codes(bucket_name)
    codes.discard(code.strip().upper())
    blob = get_bucket(bucket_name).blob("sessions/valid_codes.json")
    blob.upload_from_string(json.dumps(sorted(codes)), content_type="application/json")


def reset_session(code: str, bucket_name: str) -> None:
    """Delete all GCS data for a participant so they start completely fresh."""
    code = code.strip().upper()
    bucket = get_bucket(bucket_name)
    prefixes = [
        f"brains/{code}/",
        f"bundles/{code}/",
        f"feedback/{code}/",
        f"sessions/{code}/",
    ]
    deleted = []
    for prefix in prefixes:
        for blob in bucket.list_blobs(prefix=prefix):
            blob.delete()
            deleted.append(blob.name)

    if deleted:
        for name in deleted:
            print(f"  Deleted: {name}")
        print(f"✓ {code} reset — {len(deleted)} file(s) removed.")
    else:
        print(f"  Nothing to delete for {code} (already clean).")


def print_status(bucket_name: str) -> None:
    """Show what data exists in GCS for each known code."""
    codes = sorted(get_valid_codes(bucket_name))
    if not codes:
        print("No codes registered.")
        return

    bucket = get_bucket(bucket_name)
    for code in codes:
        has_brain  = bucket.blob(f"brains/{code}/brain.pkl").exists()
        has_bundle = bucket.blob(f"bundles/{code}/bundle.pkl").exists()
        has_pending = bucket.blob(f"sessions/{code}/pending").exists()

        fb_blob = bucket.blob(f"feedback/{code}/annotations.jsonl")
        n_feedback = 0
        if fb_blob.exists():
            n_feedback = fb_blob.download_as_text().count('\n')

        flags = []
        if has_brain:   flags.append("brain✓")
        if has_bundle:  flags.append("bundle✓")
        if has_pending: flags.append("PENDING")
        flags.append(f"{n_feedback} feedback(s)")

        print(f"  {code}: {' | '.join(flags)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", required=True)
    sub = p.add_subparsers(dest="cmd", required=True)

    add_p = sub.add_parser("add", help="Add one or more codes")
    add_p.add_argument("codes", nargs="*")
    add_p.add_argument("--participants-file", metavar="FILE",
                       help="Add all human codes from a participants.txt")

    sub.add_parser("list", help="List all valid codes")
    sub.add_parser("status", help="Show data status for each code")

    rm_p = sub.add_parser("remove", help="Remove a code (does not delete data)")
    rm_p.add_argument("code")

    rs_p = sub.add_parser("reset", help="Delete all GCS data for a code (fresh start)")
    rs_p.add_argument("codes", nargs="*")
    rs_p.add_argument("--participants-file", metavar="FILE",
                      help="Reset all human codes from a participants.txt")

    args = p.parse_args()

    if args.cmd == "add":
        codes = list(args.codes)
        if args.participants_file:
            codes += _codes_from_participants_file(args.participants_file)
        if not codes:
            print("Error: provide codes or --participants-file")
        for code in codes:
            add_valid_code(code, args.bucket)
            print(f"Added: {code.strip().upper()}")

    elif args.cmd == "list":
        codes = sorted(get_valid_codes(args.bucket))
        if codes:
            for c in codes:
                print(c)
        else:
            print("No codes found.")

    elif args.cmd == "status":
        print_status(args.bucket)

    elif args.cmd == "remove":
        remove_code(args.code, args.bucket)
        print(f"Removed from valid codes: {args.code.strip().upper()}")
        print("  (data preserved — use 'reset' to delete data too)")

    elif args.cmd == "reset":
        codes = list(args.codes)
        if args.participants_file:
            codes += _codes_from_participants_file(args.participants_file)
        if not codes:
            print("Error: provide codes or --participants-file")
        for code in codes:
            print(f"Resetting {code.strip().upper()}...")
            reset_session(code, args.bucket)


if __name__ == "__main__":
    main()
