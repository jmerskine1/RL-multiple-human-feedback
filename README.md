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

manage_codes.py   (researcher CLI)     offline_train.py  (post-study, local)
  - add / list / remove / reset codes     - downloads brains from GCS only
  - check participant status              - stack hp/hm → nTrainer=N
                                          - VI consistency estimation
                                          - train with vs. without feedback
                                          - save results.pkl locally
                                          - NO GCP compute used

         ↕──────── GCS (<your-gcs-bucket>) ────────↕

GCS layout
  bundles/<code>/bundle.pkl              pre-rendered frames
  brains/<code>/brain.pkl                Q-table + feedback counts
  feedback/<code>/annotations.jsonl      append-only feedback log
  feedback/<code>/labelled_states.json   states already seen
  sessions/<code>/pending                watcher trigger
  sessions/valid_codes.json             registered participant codes
```

### How the pipeline works

**During the study (online, per participant):**

1. `watch_bundles.py` downloads the current brain from GCS, runs a Pacman episode, selects the most informative states (active learning), and uploads a bundle to GCS.
2. The participant logs in with their code and is shown states from the bundle one at a time.
3. On each submission, Cloud Run applies the feedback to update `hp`/`hm` in the brain and saves it back to GCS. **Learning happens after every individual label.**
4. When the bundle is exhausted, the app signals GCS (`pending` marker) and shows a waiting page.
5. The watcher detects the pending signal, downloads the updated brain, runs a new episode, and uploads the next bundle — all within ~30 seconds.
6. The participant's waiting page auto-refreshes and the next batch begins.

Each participant has their own independent brain (`nTrainer=1`). The consistency estimation (Ce) runs per-submission but only self-weights that one participant against their own accumulated feedback.

**After the study (offline, all participants combined):**

7. `offline_train.py` downloads participant brains from GCS (reads only — no GCP compute), stacks their `hp`/`hm` arrays into a single agent with `nTrainer=N`, and runs the full VI consistency estimation locally. Ce[m] ∈ [0,1] is assigned per participant — reliable trainers (Ce→1) have greater influence on the final Q-table; noisy trainers (Ce→0.5) are automatically downweighted. Both a feedback condition and a pure RL baseline are trained locally and saved to `results.pkl`.

---

## Prerequisites

- Python 3.11+ with `venv`
- Docker Desktop (running)
- Google Cloud SDK (`gcloud`)
- GCP project: `<your-gcp-project>` (set in `secrets.json`)
- GCS bucket: `<your-gcs-bucket>` (set in `secrets.json`)

---

## Secrets Setup

All personal identifiers (bucket name, project ID, API keys, etc.) are stored in `secrets.json`, which is **gitignored and never committed**.

```bash
# Edit secrets.json and replace the YOUR_* placeholders with your real values
```

`secrets_loader.py` reads this file and falls back to environment variables automatically, so the same codebase works both locally (via `secrets.json`) and on Cloud Run (via env vars).

---

## One-time Setup

### 1. Authenticate locally

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project <your-gcp-project>
```

### 2. Configure Docker for GCR

```bash
gcloud auth configure-docker
```

### 3. Grant Cloud Run service account access to GCS

```bash
gsutil iam ch serviceAccount:<your-project-number>-compute@developer.gserviceaccount.com:objectAdmin \
  gs://<your-gcs-bucket>
```

> Find your project number: `gcloud projects describe <your-gcp-project> --format="value(projectNumber)"`

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
docker build --platform linux/amd64 -t gcr.io/<your-gcp-project>/<your-cloud-run-service> .

# 2. Push to Google Container Registry
docker push gcr.io/<your-gcp-project>/<your-cloud-run-service>

# 3. Deploy
gcloud run deploy <your-cloud-run-service> \
  --image gcr.io/<your-gcp-project>/<your-cloud-run-service> \
  --platform managed \
  --region <your-region> \
  --allow-unauthenticated \
  --set-env-vars GCS_BUCKET=<your-gcs-bucket>,FLASK_SECRET=<your-flask-secret>,FEEDBACK_TYPE=ordinal-feedback \
  --project <your-gcp-project>
```

**Environment variables:**

| Variable | Description |
|---|---|
| `GCS_BUCKET` | GCS bucket name |
| `FLASK_SECRET` | Secret key for signing session cookies |
| `FEEDBACK_TYPE` | `ordinal-feedback`, `binary-feedback`, or `ranked-feedback` |

**App URL:** `<your-cloud-run-url>`

> Redeploy after any code change by re-running all three commands above.

---

## Running a Study Session

### Step 1 — Add participant codes

```bash
python manage_codes.py --bucket <your-gcs-bucket> add P001 P002 P003
```

Each code becomes that participant's persistent session ID. All their data (brain, feedback, bundles) is stored under that code in GCS.

### Step 2 — Start the watcher (keep running throughout the study)

```bash
python watch_bundles.py --bucket <your-gcs-bucket>
```

The watcher polls GCS every 10 seconds. When a participant exhausts a bundle it automatically generates the next one using the updated brain. **Do not stop this while participants are active.**

### Step 3 — Send participants their credentials

```
URL:  <your-cloud-run-url>
Code: P001    (unique per participant)
```

Participants enter their code on the login page and are taken straight into the feedback interface.

---

## Managing Participants

```bash
# Add codes
python manage_codes.py --bucket <your-gcs-bucket> add P001 P002

# List all registered codes
python manage_codes.py --bucket <your-gcs-bucket> list

# Check status of all participants (brain, bundle, feedback count)
python manage_codes.py --bucket <your-gcs-bucket> status

# Remove a code (disables login, data is preserved)
python manage_codes.py --bucket <your-gcs-bucket> remove P001

# Reset a participant — deletes all their GCS data for a fresh start
python manage_codes.py --bucket <your-gcs-bucket> reset P001

# Reset multiple at once
python manage_codes.py --bucket <your-gcs-bucket> reset P001 P002 P003
```

> `remove` only disables login. `reset` deletes brain, bundle, and all feedback data entirely.

---

## Watcher Options

```bash
python watch_bundles.py \
  --bucket <your-gcs-bucket> \
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

## Offline Training (post-study)

After collecting feedback from all participants, run `offline_train.py` to aggregate their brains into a single multi-trainer model.

### Why this is needed

During the study each participant trains their own independent brain (`nTrainer=1`). The algorithm is designed to handle multiple trainers simultaneously — `hp`, `hm`, and `Ce` all have a trainer dimension. The offline step combines everyone's feedback so that:
- Consistent participants get higher Ce and more influence on the final policy
- Noisy or contradictory participants are automatically downweighted
- A single trained Q-table is produced that reflects the crowd's collective signal

### Running offline training

```bash
# All brains in bucket — omit --sessions to include everyone automatically
python offline_train.py --bucket <your-gcs-bucket> --n-episodes 200

# Specific participants only (recommended if bucket has test/stale sessions)
python offline_train.py \
    --bucket <your-gcs-bucket> \
    --sessions P001 P002 P003 \
    --n-episodes 200
```

Results are **saved locally** as `results.pkl` — no GCS upload, no cloud compute used.

### Options

| Flag | Default | Description |
|---|---|---|
| `--sessions` | *all* | Participant codes to include. **Omit to use all brains in the bucket automatically.** |
| `--n-episodes` | `100` | Training episodes per condition |
| `--max-steps` | `500` | Max steps per episode |
| `--q-init` | `zeros` | Q initialisation: `zeros` (fairest comparison) or `average` of participant tables |
| `--report-window` | `10` | Episodes to average for final performance report |
| `--save` | `results.pkl` | Local path for output file |
| `--prior-alpha` | `1.0` | Beta prior α for C estimation (flat/uninformative) |
| `--prior-beta` | `1.0` | Beta prior β for C estimation |

### What it trains

Four conditions run back-to-back for exactly `--n-episodes` episodes each:

| Condition | Description |
|---|---|
| **Individual (per participant)** | Q-learning shaped by one participant's `hp`/`hm` in isolation |
| **Combined (all participants)** | Q-learning shaped by all participants' `hp`/`hm`, weighted by Ce |
| **Baseline** | Fresh agent — pure Q-learning, no human signal |

### Plotting the results

Once `results.pkl` exists, generate the learning curve plot:

```bash
python analyse_results.py results.pkl

# Custom smoothing window and output path
python analyse_results.py results.pkl --window 30 --save my_plot.png
```

This produces a three-panel dark-theme figure (`learning_curves.png` by default):
- **Top**: All learning curves — each participant (thin, coloured, Ce in legend), combined (gold), baseline (blue)
- **Bottom left**: Final performance bar chart across all conditions
- **Bottom right**: Ce consistency per participant with reliability thresholds

### Example output

```
── Running C-estimation ──

══════════════════════════════════════════
  Rank  Session            Ce   Signal
──────────────────────────────────────────
  1     P001            0.834   ✓ reliable
  2     P002            0.761   ✓ reliable
  3     P003            0.509   ✗ noisy
══════════════════════════════════════════

── Training P001 individually (200 episodes) ──
── Training P002 individually (200 episodes) ──
── Training P003 individually (200 episodes) ──

── Training WITH feedback — combined (200 episodes) ──
  [feedback]   ep   20/200  reward=  12.40  steps=87    Ce=[0.841, 0.774, 0.511]
  ...

── Training WITHOUT feedback — baseline (200 episodes) ──
  [baseline]   ep   20/200  reward=   4.20  steps=312
  ...

── Performance comparison (mean reward, last 20 episodes) ──
  With    human feedback:   18.60
  Without human feedback:    9.30
  Δ (feedback − baseline):  +9.30  (+100.0%)
```

### Using the results file

```python
import pickle
r = pickle.load(open("results.pkl", "rb"))

r["sessions"]              # list of participant codes included
r["Ce"]                    # consistency estimate per participant (combined model)
r["Ce_individual"]         # {session: Ce scalar} from individual models
r["rewards_feedback"]      # list of reward per episode — combined with feedback
r["rewards_baseline"]      # list of reward per episode — no feedback
r["rewards_individual"]    # {session: [rewards]} — per-participant conditions
r["Q_feedback"]            # final Q-table from combined feedback condition
r["Q_baseline"]            # final Q-table from baseline condition
```

### Interpreting Ce values

| Ce range | Meaning |
|---|---|
| 0.75 – 1.0 | Reliable — feedback consistently matched the agent's learned policy |
| 0.55 – 0.75 | Moderate — reasonable signal but some inconsistency |
| 0.5 – 0.55 | Noisy — feedback no better than random; effectively ignored |

Ce → 0.5 does not mean the participant gave bad feedback — it may mean they saw too few states for a reliable estimate, or their mental model diverges from the RL agent's learned values.

---

## Troubleshooting

### Internal Server Error on `/submit`

Check logs:
```bash
gcloud run services logs read <your-cloud-run-service> --region <your-region> \
  --project <your-gcp-project> --limit 50
```

### Participant stuck on waiting page

The watcher is not running or crashed. Restart it — it will immediately catch up on any pending sessions.

### `PERMISSION_DENIED` accessing GCS

```bash
gsutil iam ch serviceAccount:<your-project-number>-compute@developer.gserviceaccount.com:objectAdmin \
  gs://<your-gcs-bucket>
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
| `offline_train.py` | Local | Post-study aggregation — downloads brains from GCS, trains locally, saves `results.pkl` |
| `analyse_results.py` | Local | Plots learning curves and consistency report from `results.pkl` |
| `trainer.py` | Both | Agent wrapper — brain load/save |
| `agent.py` | Both | RL agent (tabQL with VI-based C estimation) |
| `secrets_loader.py` | Both | Reads `secrets.json` with env var fallback — never commit `secrets.json` |
| `Dockerfile` | — | Cloud Run container definition |
| `requirements_gcp.txt` | — | Cloud Run Python deps (no torch/matplotlib) |
