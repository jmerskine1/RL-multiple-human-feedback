"""
gcs_utils.py — Cloud Storage helpers shared by bundle_generator.py (local)
and main_gcp.py (GCP).

All heavy data lives in GCS; the Flask app session cookie only stores a
session hash and a queue pointer.

GCS layout
----------
gs://<BUCKET>/
  bundles/<hash>/bundle.pkl          # pre-rendered frames (written locally)
  brains/<hash>/brain.pkl            # Q/Ce/hp/hm   (written by GCP on submit)
  feedback/<hash>/annotations.jsonl  # append-only log (written by GCP)
  feedback/<hash>/labelled_states.json  # set of already-labelled obs IDs
"""

import pickle
import json
from datetime import datetime

from google.cloud import storage


# ── bucket handle (cached per process) ───────────────────────────────────────

_client = None

def _get_client():
    global _client
    if _client is None:
        _client = storage.Client()
    return _client

def get_bucket(bucket_name: str):
    return _get_client().bucket(bucket_name)


# ── bundles ───────────────────────────────────────────────────────────────────

def upload_bundle(session_hash: str, bundle: dict, bucket_name: str) -> None:
    blob = get_bucket(bucket_name).blob(f"bundles/{session_hash}/bundle.pkl")
    blob.upload_from_string(pickle.dumps(bundle), content_type="application/octet-stream")
    print(f"[gcs] bundle uploaded → gs://{bucket_name}/bundles/{session_hash}/bundle.pkl")

def download_bundle(session_hash: str, bucket_name: str) -> dict | None:
    blob = get_bucket(bucket_name).blob(f"bundles/{session_hash}/bundle.pkl")
    if not blob.exists():
        return None
    return pickle.loads(blob.download_as_bytes())

def bundle_exists(session_hash: str, bucket_name: str) -> bool:
    return get_bucket(bucket_name).blob(f"bundles/{session_hash}/bundle.pkl").exists()


# ── brains ────────────────────────────────────────────────────────────────────

def upload_brain(session_hash: str, brain_data: dict, bucket_name: str) -> None:
    blob = get_bucket(bucket_name).blob(f"brains/{session_hash}/brain.pkl")
    blob.upload_from_string(pickle.dumps(brain_data), content_type="application/octet-stream")

def download_brain(session_hash: str, bucket_name: str) -> dict | None:
    blob = get_bucket(bucket_name).blob(f"brains/{session_hash}/brain.pkl")
    if not blob.exists():
        return None
    return pickle.loads(blob.download_as_bytes())

def brain_exists(session_hash: str, bucket_name: str) -> bool:
    return get_bucket(bucket_name).blob(f"brains/{session_hash}/brain.pkl").exists()


# ── labelled state memory ─────────────────────────────────────────────────────

def get_labelled_states(session_hash: str, bucket_name: str) -> set:
    blob = get_bucket(bucket_name).blob(f"feedback/{session_hash}/labelled_states.json")
    if not blob.exists():
        return set()
    return set(json.loads(blob.download_as_text()))

def add_labelled_state(session_hash: str, obs: int, bucket_name: str) -> None:
    states = get_labelled_states(session_hash, bucket_name)
    states.add(int(obs))
    blob = get_bucket(bucket_name).blob(f"feedback/{session_hash}/labelled_states.json")
    blob.upload_from_string(json.dumps(list(states)), content_type="application/json")


# ── feedback log ──────────────────────────────────────────────────────────────

def append_annotation(session_hash: str, record: dict, bucket_name: str) -> None:
    """Append one JSON record to the session's annotation log."""
    blob = get_bucket(bucket_name).blob(f"feedback/{session_hash}/annotations.jsonl")
    existing = blob.download_as_text() if blob.exists() else ""
    record["timestamp"] = datetime.utcnow().isoformat()
    blob.upload_from_string(existing + json.dumps(record) + "\n",
                            content_type="application/json")
