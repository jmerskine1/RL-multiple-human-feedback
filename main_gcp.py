"""
main_gcp.py — GCP-optimised Flask app for Option-B hybrid deployment.

What's different from main.py
------------------------------
* NO matplotlib — frames are pre-rendered by bundle_generator.py (local).
* NO Redis / KVSession — sessions use signed cookies (tiny payload).
* NO episode simulation on startup — loads pre-rendered bundle from GCS.
* Confidence plot rendered client-side with Chart.js (no server matplotlib).
* Brain updates (numpy only) happen on /submit and are persisted to GCS.
* When a bundle is exhausted the participant sees a friendly "processing" page
  and the researcher runs bundle_generator.py to upload the next batch.

GCS layout (managed by gcs_utils.py)
--------------------------------------
  bundles/<hash>/bundle.pkl          written by bundle_generator.py (local)
  brains/<hash>/brain.pkl            read + written by this app
  feedback/<hash>/annotations.jsonl  append-only feedback log
  feedback/<hash>/labelled_states.json
"""

import os
import json
import pickle

from flask import Flask, render_template, request, session, jsonify, redirect, url_for
import numpy as np

from feedback import Feedback
from trainer import PacmanTrainer
from gcs_utils import (
    download_bundle, download_brain, upload_brain,
    append_annotation, add_labelled_state, write_pending, is_valid_code,
)

# ── App setup ──────────────────────────────────────────────────────────────────
from secrets_loader import secret

app = Flask(__name__)
app.secret_key = secret("flask_secret", env_var="FLASK_SECRET", default="change-me-in-production")

GCS_BUCKET = secret("gcs_bucket", env_var="GCS_BUCKET")  # set via secrets.json or env var

# ── Config ─────────────────────────────────────────────────────────────────────
FEEDBACK_TYPE   = os.environ.get("FEEDBACK_TYPE", "ordinal-feedback")
ACTIVE_LEARNING_MODE = "count"

_FEEDBACK_TEMPLATES = {
    "binary-feedback":  "index.html",
    "ranked-feedback":  "index_ranked.html",
    "ordinal-feedback": "index_ordinal.html",
}

def get_template():
    return _FEEDBACK_TEMPLATES.get(FEEDBACK_TYPE, "index.html")

# ── In-memory bundle cache ─────────────────────────────────────────────────────
# Bundles are large (pre-rendered base64 images). We cache them in process
# memory so Cloud Run doesn't re-download from GCS on every request.
# With --workers=1 in gunicorn a single instance handles all requests for the
# session, keeping the cache coherent.
_bundle_cache: dict[str, dict] = {}

def _load_bundle(session_hash: str) -> dict | None:
    if session_hash not in _bundle_cache:
        bundle = download_bundle(session_hash, GCS_BUCKET)
        if bundle:
            _bundle_cache[session_hash] = bundle
    return _bundle_cache.get(session_hash)

def _evict_bundle(session_hash: str) -> None:
    """Remove a bundle from cache (e.g. after it's been fully consumed)."""
    _bundle_cache.pop(session_hash, None)

    # Also delete from GCS so bundle_generator knows a new one is needed
    from gcs_utils import get_bucket
    blob = get_bucket(GCS_BUCKET).blob(f"bundles/{session_hash}/bundle.pkl")
    if blob.exists():
        blob.delete()


# ── Session helpers ────────────────────────────────────────────────────────────
# The cookie session stores only lightweight data:
#   session_hash       — participant code (set at login)
#   current_queue_idx  — position within selected_indices
#   Ce_list            — list of floats (grows with feedback, ~8B each)

def _logged_in() -> bool:
    # authenticated flag is only set by the login route — prevents old cookies bypassing login
    return "session_hash" in session and session.get("authenticated", False)

def _ensure_session():
    """Redirect to login if participant hasn't entered their code."""
    if not _logged_in():
        return redirect(url_for("login"))
    session.setdefault("current_queue_idx", 0)
    session.setdefault("Ce_list", [0.5])
    return None

def _sync_queue_idx(bundle: dict) -> int:
    """
    Return a guaranteed-valid current_queue_idx for this bundle.

    Guards against two crash scenarios:
      1. Stale cookie — participant went back in browser history after bundle
         rotation; their cookie still carries an index from the old (longer)
         bundle.
      2. Bundle size changed — the watcher generated a shorter next batch
         (fewer unlabelled states left), making the old index out of range.

    Also resets the index when the bundle itself has changed (detected via the
    length stored in the session at the time the bundle was first seen).
    """
    queue_len   = len(bundle["selected_indices"])
    stored_len  = session.get("bundle_len")
    q           = session.get("current_queue_idx", 0)

    # Detect bundle rotation: if the stored length doesn't match the current
    # bundle, the participant has moved onto a new bundle — reset to position 0.
    if stored_len != queue_len:
        q = 0
        session["bundle_len"] = queue_len

    # Hard clamp — should never be needed after the above, but prevents a crash
    # even if session state is corrupt for any other reason.
    if q < 0 or q >= queue_len:
        q = 0

    session["current_queue_idx"] = q
    return q


def _render(bundle: dict, **extra):
    q        = _sync_queue_idx(bundle)
    idx      = bundle["selected_indices"][q]
    img2     = bundle["all_plots"][idx]
    valid_mv = bundle["all_valid_moves"][idx]
    queue_pos = q + 1
    queue_len = len(bundle["selected_indices"])
    return render_template(
        get_template(),
        img2=img2,
        img1="",                   # not used in GCP templates
        graph="",                  # confidence plot done client-side
        ce_list=json.dumps(session.get("Ce_list", [0.5])),
        queue_pos=queue_pos,
        queue_len=queue_len,
        valid_moves=valid_mv,
        session_status=bundle["all_status"][idx],
        **extra,
    )


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        code = request.form.get("code", "").strip().upper()
        if is_valid_code(code, GCS_BUCKET):
            session.clear()
            session["session_hash"]      = code
            session["authenticated"]     = True
            session["current_queue_idx"] = 0
            session["Ce_list"]           = [0.5]
            return redirect(url_for("index"))
        return render_template("login.html", error="Invalid code — please check and try again.")
    return render_template("login.html", error=None)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
def index():
    redir = _ensure_session()
    if redir:
        return redir
    session["current_queue_idx"] = 0

    bundle = _load_bundle(session["session_hash"])
    if not bundle:
        write_pending(session["session_hash"], GCS_BUCKET)
        return render_template("waiting.html",
                               session_hash=session["session_hash"])

    return _render(bundle)


@app.route("/next", methods=["POST"])
def nextFrame():
    redir = _ensure_session()
    if redir:
        return redir
    bundle = _load_bundle(session["session_hash"])
    if not bundle:
        return render_template("waiting.html",
                               session_hash=session["session_hash"])

    q = _sync_queue_idx(bundle)
    if q + 1 < len(bundle["selected_indices"]):
        session["current_queue_idx"] = q + 1

    return _render(bundle)


@app.route("/previous", methods=["POST"])
def previousFrame():
    redir = _ensure_session()
    if redir:
        return redir
    bundle = _load_bundle(session["session_hash"])
    if not bundle:
        return render_template("waiting.html",
                               session_hash=session["session_hash"])

    q = _sync_queue_idx(bundle)
    if q > 0:
        session["current_queue_idx"] = q - 1

    return _render(bundle)


@app.route("/submit", methods=["POST"])
def submit():
    redir = _ensure_session()
    if redir:
        return redir
    bundle = _load_bundle(session["session_hash"])
    if not bundle:
        return render_template("waiting.html",
                               session_hash=session["session_hash"])

    # ── 1. Rebuild Feedback object from human input ───────────────────────────
    data          = request.get_json()
    arrow_dict    = {"arrowup": "n", "arrowdown": "s", "arrowleft": "w", "arrowright": "e"}
    idx           = bundle["selected_indices"][_sync_queue_idx(bundle)]
    obs           = bundle["all_obs"][idx]
    taken_action  = bundle["all_actions"][idx]
    valid_moves   = bundle["all_valid_moves"][idx]          # [n, s, e, w]
    action_list   = ["n", "s", "e", "w"]
    invalid_idx   = [i for i, v in enumerate(valid_moves) if not v]
    n_actions     = len(action_list)

    if FEEDBACK_TYPE == "binary-feedback":
        feedback_string = arrow_dict[data["arrow"]]
        action = action_list.index(feedback_string)
        if action == taken_action:
            fb_obj = Feedback(state=obs, good_actions=[action], conf_good_actions=1.0)
        else:
            fb_obj = Feedback(state=obs, bad_actions=[action], conf_bad_actions=1.0)
        db_feedback = feedback_string

    elif FEEDBACK_TYPE == "ranked-feedback":
        ranking = data["ranking"]
        Q_syn = np.zeros(n_actions)
        for rank_0, arrow_str in enumerate(ranking):
            Q_syn[action_list.index(arrow_dict[arrow_str])] = n_actions - 1 - rank_0
        for i in invalid_idx:
            Q_syn[i] = -1
        sorted_idx = np.argsort(-Q_syn).tolist()
        fb_obj = Feedback(state=obs,
                          good_actions=sorted_idx[:2], bad_actions=sorted_idx[2:],
                          conf_good_actions=1.0, conf_bad_actions=1.0)
        db_feedback = json.dumps(ranking)

    elif FEEDBACK_TYPE == "ordinal-feedback":
        values = data["values"]
        Q_syn = np.full(n_actions, 0.5)   # 0.5 = neutral, no signal to agent
        for arrow_str, val in values.items():
            Q_syn[action_list.index(arrow_dict[arrow_str])] = float(val)
        for i in invalid_idx:
            Q_syn[i] = 0.5                # invalid moves also get no signal
        fb_obj = Feedback(state=obs, good_actions=Q_syn.tolist(), conf_good_actions=1.0)
        db_feedback = json.dumps(values)

    else:
        return jsonify({"error": "unknown FEEDBACK_TYPE"}), 400

    # ── 2. Load brain, apply feedback update (pure numpy) ────────────────────
    trainer = PacmanTrainer(
        algID="tabQL_Cest_vi_t2",
        env_size=bundle.get("env_size", "small"),
        active_feedback_type=ACTIVE_LEARNING_MODE,
    )
    brain = download_brain(session["session_hash"], GCS_BUCKET)
    if brain:
        trainer.load_brain(brain)
    else:
        # Fresh brain — trigger Q/hp/hm initialisation via a dummy act() call
        # (the init block inside tabQL_Cest_vi only runs when Q is None).
        # prev_obs stays None so no Q-update or feedback side-effects occur.
        trainer.agent.act(taken_action, obs, 0, False, [[]], 0.5, update_Cest=False)

    # Apply human feedback directly to hp/hm.
    # We bypass act() here because act() guards _collect_feedback behind
    # `if self.prev_obs is not None`, which is always None on a fresh request.
    trainer.agent._collect_feedback([[fb_obj]])

    # Re-estimate consistency from updated hp/hm.
    # Reset prev_obs first so the act() call below doesn't trigger a spurious
    # Q-learning self-loop update — only the EM loop at the end needs to run.
    trainer.agent.prev_obs = None
    trainer.agent.act(taken_action, obs, 0, False, [[]], 0.5, update_Cest=True)

    # ── 3. Persist updated brain and feedback log ─────────────────────────────
    upload_brain(session["session_hash"], trainer.get_brain(), GCS_BUCKET)
    add_labelled_state(session["session_hash"], int(obs), GCS_BUCKET)
    append_annotation(session["session_hash"], {
        "obs":       int(obs),
        "action":    action_list[taken_action],
        "feedback":  db_feedback,
        "ce":        float(trainer.agent.Ce[0]),
    }, GCS_BUCKET)

    # ── 4. Update Ce_list in session cookie ──────────────────────────────────
    ce_list = session.get("Ce_list", [0.5])
    ce_list.append(float(trainer.agent.Ce[0]))
    session["Ce_list"] = ce_list

    # ── 5. Advance queue or request new bundle ────────────────────────────────
    q = _sync_queue_idx(bundle)
    if q + 1 < len(bundle["selected_indices"]):
        session["current_queue_idx"] = q + 1
        return _render(bundle)
    else:
        # Bundle exhausted — evict cache + GCS copy, signal watcher to generate next
        _evict_bundle(session["session_hash"])
        write_pending(session["session_hash"], GCS_BUCKET)
        return render_template("waiting.html",
                               session_hash=session["session_hash"])


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
