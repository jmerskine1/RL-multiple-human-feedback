# RL-multiple-human-feedback

Adapted version of repo from [arXiv](https://arxiv.org/abs/2111.08596) to test with collection from real humans.

---

## Architecture

```
LOCAL MACHINE                          GCP CLOUD RUN
─────────────────────────────────      ──────────────────────────────────
watch_bundles.py  (keep running)       main_gcp.py  (Docker → Cloud Run)
  - simulate Pacman episodes             - login page (participant codes)
  - render frames with matplotlib        - serve pre-rendered states
  - active learning state selection      - collect ordinal feedback
  - upload bundle → GCS                  - update brain (numpy only)
                                         - write brain → GCS
                                         - signal pending → GCS

manage_codes.py   (researcher CLI)
  - add / list / remove / reset codes
  - check participant status

         ↕──────── GCS (jerskine_human_feedback) ────────↕

GCS layout
  bundles/<code>/bundle.pkl              pre-rendered frames
  brains/<code>/brain.pkl                Q-table + feedback counts
  feedback/<code>/annotations.jsonl      append-only feedback log
  feedback/<code>/labelled_states.json   states already seen
  sessions/<code>/pending                watcher trigger
  sessions/valid_codes.json             registered participant codes
```

### How the pipeline works

1. `watch_bundles.py` downloads the current brain from GCS, runs a Pacman episode, selects the most informative states (active learning), and uploads a bundle to GCS.
2. The participant logs in with their code and is shown states from the bundle one at a time.
3. On each submission, Cloud Run applies the feedback to update the brain and saves it back to GCS. **Learning happens after every individual label.**
4. When the bundle is exhausted, the app signals GCS (`pending` marker) and shows a waiting page.
5. The watcher detects the pending signal, downloads the updated brain, runs a new episode, and uploads the next bundle — all within ~30 seconds.
6. The participant's waiting page auto-refreshes and the next batch begins.

---

## Prerequisites

- Python 3.11+ with `venv`
- Docker Desktop (running)
- Google Cloud SDK (`gcloud`)
- GCP project: `pacman-data-collection`
- GCS bucket: `jerskine_human_feedback`

---

## One-time Setup

### 1. Authenticate locally

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project pacman-data-collection
```

### 2. Configure Docker for GCR

```bash
gcloud auth configure-docker
```

### 3. Grant Cloud Run service account access to GCS

```bash
gsutil iam ch serviceAccount:323412919890-compute@developer.gserviceaccount.com:objectAdmin \
  gs://jerskine_human_feedback
```

### 4. Install local Python dependencies

```bash
pip install -r requirements.txt        # full local deps (matplotlib, torch, etc.)
pip install google-cloud-storage       # GCS access for local scripts
```

---

## Deploying to Cloud Run

Run from the `RL-multiple-human-feedback/` directory.

```bash
# 1. Build for linux/amd64 (required — Mac is arm64)
docker build --platform linux/amd64 -t gcr.io/pacman-data-collection/pacman-feedback .

# 2. Push to Google Container Registry
docker push gcr.io/pacman-data-collection/pacman-feedback

# 3. Deploy
gcloud run deploy pacman-feedback \
  --image gcr.io/pacman-data-collection/pacman-feedback \
  --platform managed \
  --region europe-west2 \
  --allow-unauthenticated \
  --set-env-vars GCS_BUCKET=jerskine_human_feedback,FLASK_SECRET='6181',FEEDBACK_TYPE=ordinal-feedback \
  --project pacman-data-collection
```

**Environment variables:**

| Variable | Description |
|---|---|
| `GCS_BUCKET` | GCS bucket name |
| `FLASK_SECRET` | Secret key for signing session cookies |
| `FEEDBACK_TYPE` | `ordinal-feedback`, `binary-feedback`, or `ranked-feedback` |

**App URL:** `https://pacman-feedback-323412919890.europe-west2.run.app`

> Redeploy after any code change by re-running all three commands above.

---

## Running a Study Session

### Step 1 — Add participant codes

```bash
python manage_codes.py --bucket jerskine_human_feedback add P001 P002 P003
```

Each code becomes that participant's persistent session ID. All their data (brain, feedback, bundles) is stored under that code in GCS.

### Step 2 — Start the watcher (keep running throughout the study)

```bash
python watch_bundles.py --bucket jerskine_human_feedback
```

The watcher polls GCS every 10 seconds. When a participant exhausts a bundle it automatically generates the next one using the updated brain. **Do not stop this while participants are active.**

### Step 3 — Send participants their credentials

```
URL:  https://pacman-feedback-323412919890.europe-west2.run.app
Code: P001    (unique per participant)
```

Participants enter their code on the login page and are taken straight into the feedback interface.

---

## Managing Participants

```bash
# Add codes
python manage_codes.py --bucket jerskine_human_feedback add P001 P002

# List all registered codes
python manage_codes.py --bucket jerskine_human_feedback list

# Check status of all participants (brain, bundle, feedback count)
python manage_codes.py --bucket jerskine_human_feedback status

# Remove a code (disables login, data is preserved)
python manage_codes.py --bucket jerskine_human_feedback remove P001

# Reset a participant — deletes all their GCS data for a fresh start
python manage_codes.py --bucket jerskine_human_feedback reset P001

# Reset multiple at once
python manage_codes.py --bucket jerskine_human_feedback reset P001 P002 P003
```

> `remove` only disables login. `reset` deletes brain, bundle, and all feedback data entirely.

---

## Watcher Options

```bash
python watch_bundles.py \
  --bucket jerskine_human_feedback \
  --interval 10 \        # poll interval in seconds (default: 10)
  --n-feedbacks 10 \     # states per bundle (default: 10)
  --mode count \         # active learning: count or entropy
  --env-size small \     # small, medium, or medium_sparse
  --max-steps 500        # max episode length
```

---

## Feedback Types

| Type | Description |
|---|---|
| `ordinal-feedback` | Hold each arrow to score it 0→1. Default 0.5 = no signal. |
| `ranked-feedback` | Click arrows to rank best→worst. |
| `binary-feedback` | Select the single best action. |

Set via the `FEEDBACK_TYPE` environment variable at deploy time.

### Ordinal scoring details

- Score `> 0.5` → positive signal (confidence = distance from 0.5, scaled to 0–1)
- Score `< 0.5` → negative signal
- Score `= 0.5` → no signal (default for unscored and invalid actions)
- Invalid moves are greyed out and always produce no signal

---

## Troubleshooting

### Internal Server Error on `/submit`

Check logs:
```bash
gcloud run services logs read pacman-feedback --region europe-west2 \
  --project pacman-data-collection --limit 50
```

### Participant stuck on waiting page

The watcher is not running or crashed. Restart it — it will immediately catch up on any pending sessions.

### `PERMISSION_DENIED` accessing GCS

```bash
gsutil iam ch serviceAccount:323412919890-compute@developer.gserviceaccount.com:objectAdmin \
  gs://jerskine_human_feedback
```

### Docker daemon not running

Open Docker Desktop and wait for the menu bar whale icon to stop animating.

### `ModuleNotFoundError: google`

```bash
pip install google-cloud-storage
```

### `DefaultCredentialsError`

```bash
gcloud auth application-default login
```

---

## Key Files

| File | Where it runs | Purpose |
|---|---|---|
| `main_gcp.py` | Cloud Run | Flask app — login, feedback UI, brain updates |
| `gcs_utils.py` | Both | GCS read/write helpers |
| `bundle_generator.py` | Local | Episode simulation and bundle creation |
| `watch_bundles.py` | Local | Auto-generates bundles during study sessions |
| `manage_codes.py` | Local | Researcher CLI for participant management |
| `trainer.py` | Both | Agent wrapper — brain load/save |
| `agent.py` | Both | RL agent (tabQL with VI-based C estimation) |
| `Dockerfile` | — | Cloud Run container definition |
| `requirements_gcp.txt` | — | Cloud Run Python deps (no torch/matplotlib) |
