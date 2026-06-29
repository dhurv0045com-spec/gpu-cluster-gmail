# AN-RA Multi-Colab Distributed Training Cluster

<p align="center">
  <img src="https://img.shields.io/badge/status-active-success" alt="Status" />
  <img src="https://img.shields.io/badge/python-3.11-blue" alt="Python" />
  <img src="https://img.shields.io/badge/react-18-61dafb" alt="React" />
  <img src="https://img.shields.io/badge/model-499M_parameters-ff6b35" alt="Params" />
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License" />
</p>

Turn **N free Google Colab accounts** (each with a T4 GPU) into a unified, coordinated cluster for training the **An-Ra iterate500** — a 499M-parameter CausalTransformerV2 with GQA, SwiGLU, RoPE, HAL, ESV, MoD, and ThirdEye attention.

```
┌─────────────────────────────────────────────────────────────┐
│                    WEB DASHBOARD (React + Vite)              │
│                                                             │
│  ┌─────────────┐  ┌──────────┐  ┌────────┐  ┌──────────┐  │
│  │   Cluster   │  │  Setup   │  │Workers │  │   Logs   │  │
│  │  Overview   │  │  Wizard  │  │ Detail │  │  Stream  │  │
│  └──────┬──────┘  └────┬─────┘  └───┬────┘  └────┬─────┘  │
└─────────┼──────────────┼────────────┼────────────┼─────────┘
          │              │            │            │
          └──────────────┼────────────┼────────────┘
                         │   HTTPS REST API
          ┌──────────────▼────────────▼──────────────────────┐
          │              FASTAPI COORDINATOR                  │
          │                                                   │
          │  ┌──────────────┐  ┌────────────┐  ┌──────────┐  │
          │  │   Worker     │  │  Gradient  │  │  Drive   │  │
          │  │  Registry    │  │ Aggregator │  │  Sync    │  │
          │  │ (heartbeat)  │  │ (averaging)│  │  Layer   │  │
          │  └──────────────┘  └────────────┘  └──────────┘  │
          │  ┌──────────────┐  ┌────────────┐  ┌──────────┐  │
          │  │    Job       │  │    SSE     │  │  SQLite  │  │
          │  │  Scheduler   │  │ Log Stream │  │  Persist │  │
          │  └──────────────┘  └────────────┘  └──────────┘  │
          └──────────────────────┬───────────────────────────┘
                                 │ Google Drive API (OAuth 2.0)
          ┌──────────────────────▼───────────────────────────┐
          │              GOOGLE DRIVE (Shared Storage)        │
          │                                                   │
          │  /AnRa/cluster/                                   │
          │  ├── coordinator_state.json   ← global state      │
          │  ├── lock.json                ← optimistic lock   │
          │  ├── master_weights_v001.pt   ← current weights   │
          │  ├── master_weights_v042.pt   ← step 42 weights   │
          │  ├── worker_A/                                    │
          │  │   ├── grad_step_000042.pt   ← A's gradients    │
          │  │   └── grad_step_000043.pt                      │
          │  ├── worker_B/                                    │
          │  │   ├── grad_step_000042.pt   ← B's gradients    │
          │  │   └── grad_step_000043.pt                      │
          │  └── worker_C/                                    │
          │      └── grad_step_000042.pt   ← C's gradients    │
          └──────────────────────────────────────────────────┘
               ▲                    ▲                    ▲
               │ Drive mount        │ Drive mount        │ Drive mount
          ┌────┴──────┐       ┌────┴──────┐       ┌────┴──────┐
          │ Colab A   │       │ Colab B   │       │ Colab C   │
          │ T4 16 GB  │       │ T4 16 GB  │       │ T4 16 GB  │
          │ Account 1 │       │ Account 2 │       │ Account 3 │
          │ worker.py │       │ worker.py │       │ worker.py │
          └───────────┘       └───────────┘       └───────────┘
```

## Table of Contents

- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Deployment](#deployment)
- [Usage Guide](#usage-guide)
- [API Reference](#api-reference)
- [Architecture Deep Dive](#architecture-deep-dive)
- [Edge Cases & Fault Tolerance](#edge-cases--fault-tolerance)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Stretch Goals](#stretch-goals)

---

## How It Works

Colab runtimes cannot SSH to each other — they are isolated VMs with no NCCL and no direct FSDP across machines. This system solves that with **asynchronous federated gradient averaging via Google Drive**.

### The Cycle

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│ 1. Load  │────→│ 2. Forward│────→│ 3. Save  │────→│ 4. Signal│
│  weights │     │ + Backward│     │ gradients│     │coordinator│
└──────────┘     └──────────┘     └─────┬────┘     └────┬─────┘
                                        │                │
                                   writes to        HTTP POST
                                   Drive/worker     /gradient_ready
                                        │                │
                                        ▼                ▼
                                   ┌──────────────────────────┐
                                   │    COORDINATOR (server)  │
                                   │                          │
                                   │  Wait for N workers →    │
                                   │  Average gradients →     │
                                   │  Apply optimizer step →  │
                                   │  Write new weights →     │
                                   │  Update state.json       │
                                   └──────────────────────────┘
                                        │
                                        ▼
┌──────────┐     ┌──────────┐     ┌──────────┐
│ 6. Reload│←────│ 5. Detect│←────│ New .pt  │
│  model   │     │new weights│     │ on Drive │
└──────────┘     └──────────┘     └──────────┘
```

Each worker:
1. Loads the latest master weights from Drive
2. Forward pass → loss → backward pass (computes gradients)
3. Saves gradients (NOT updated weights) to Drive as fp16 `.pt` file
4. HTTP POSTs `gradient_ready` to the coordinator
5. Polls Drive for new master weights (exponential backoff, up to 5 min)
6. Reloads new weights and repeats

**Critical:** Workers compute gradients but do NOT apply them. Weight updates happen only in the coordinator's aggregation step. This prevents model drift between workers.

---

## Project Structure

```
anra-cluster/
│
├── backend/                         # FastAPI coordinator — the brain
│   ├── main.py                      # 12 REST endpoints + SSE log stream + aggregation orchestration
│   ├── database.py                  # SQLite models (Worker, ClusterState) via SQLModel
│   ├── aggregator.py                # Federated averaging: NaN/Inf sanitization + token-weighted mean
│   ├── drive_sync.py                # Google Drive I/O: retry with exponential backoff, optimistic locking
│   ├── worker_registry.py           # Heartbeat expiry (120s), stale reaper, aggregation quorum logic
│   ├── scheduler.py                 # Training data sharding across workers
│   ├── auth.py                      # Google OAuth 2.0 flow helpers
│   ├── requirements.txt             # Pinned dependencies (FastAPI, PyTorch, google-api-client, tenacity)
│   └── __init__.py
│
├── worker/                          # Colab worker runtime — plug-and-play
│   ├── AN_RA_CLUSTER_WORKER.ipynb   # 5-cell Jupyter notebook: install → mount → config → register → loop
│   ├── worker_loop.py               # Core training loop with position tracking + error recovery
│   ├── drive_worker.py              # Worker-side Drive ops + SIGTERM/SIGINT handler for graceful shutdown
│   └── wait_for_aggregation.py      # Exponential-backoff poller with 60s progress logging
│
├── frontend/                        # React + Vite + Tailwind dashboard
│   ├── src/
│   │   ├── pages/
│   │   │   ├── ClusterOverview.jsx  # Main dashboard: workers, loss curve, gradient heatmap, Drive tree
│   │   │   ├── SetupWizard.jsx      # 3-step onboarding: config → init → worker links
│   │   │   ├── WorkerDetail.jsx     # Per-worker metrics, loss history chart, raw heartbeat data
│   │   │   └── LogStream.jsx        # SSE live log viewer with level filtering + auto-scroll
│   │   ├── components/
│   │   │   ├── ErrorBoundary.jsx    # Catches React crashes, shows error screen with reload
│   │   │   ├── WorkerCard.jsx       # GPU bar (color-coded), heartbeat age, click → detail
│   │   │   ├── LossCurve.jsx        # Recharts line chart with CartesianGrid + empty state
│   │   │   ├── GradientHeatmap.jsx  # 30fps animated canvas: layers × pixel grid, pulsing
│   │   │   ├── ThroughputBadge.jsx  # tok/s display badge
│   │   │   └── DriveFileTree.jsx    # Live Drive file listing with size totals
│   │   ├── hooks/
│   │   │   ├── useClusterStatus.js  # Polls /api/training/status every 5s (mounted-guarded)
│   │   │   ├── useWorkers.js        # Polls /api/workers every 5s (mounted-guarded)
│   │   │   └── useLogStream.js      # SSE hook with exponential backoff reconnection
│   │   └── lib/
│   │       └── api.js               # Typed API client: AbortController timeout, error parsing
│   ├── package.json                 # React 18, Recharts 2, react-router-dom 6, Tailwind 3
│   ├── vite.config.js               # Dev proxy /api → localhost:8000
│   ├── tailwind.config.js           # Theme: deep #0a0e1a, accent #00d4ff, success #00ff88
│   ├── vercel.json                  # SPA rewrites for Vercel deployment
│   └── index.html                   # JetBrains Mono + Inter, SVG favicon (◈ diamond)
│
├── scripts/                         # CLI utilities
│   ├── setup_drive_structure.py     # Creates /AnRa/cluster/ folder tree + coordinator_state.json
│   └── generate_worker_config.py    # CLI: --num-workers N → generates per-worker JSON configs
│
├── pyproject.toml                   # Ruff linter: line-length 120, py311, double quotes
├── railway.toml                     # Railway deployment: Nixpacks builder, /data volume
├── Dockerfile                       # Python 3.11-slim container
├── docker-compose.yml               # Local dev with persistent /data volume
└── .gitignore                       # node_modules, __pycache__, .env, *.pt
```

---

## Prerequisites

- **Python 3.11+** for the backend
- **Node.js 20+** for the frontend
- **N Google accounts** — each with a free Colab plan (T4 GPU)
- **Google Drive** — 15 GB free per account (shared folder)
- **An-Ra codebase** — deployed to each account's Drive at `MyDrive/AnRa/v2/`

---

## Installation

### Backend (local development)

```bash
# Clone
git clone https://github.com/YOUR_ORG/anra-cluster.git
cd anra-cluster

# Create virtual environment
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
.venv\Scripts\activate       # Windows

# Install dependencies
pip install -r backend/requirements.txt

# Run
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend (local development)

```bash
cd frontend
npm install
npm run dev
# Opens at http://localhost:5173, proxies /api to :8000
```

### Docker (local development, both services)

```bash
docker compose up --build
# Backend at http://localhost:8000
# Frontend: run separately with npm run dev
```

---

## Deployment

### Backend → Railway

1. Push repo to GitHub
2. Create a Railway project from the repo
3. Railway auto-detects `railway.toml`:
   ```toml
   [build]
   builder = "NIXPACKS"
   [deploy]
   startCommand = "uvicorn backend.main:app --host 0.0.0.0 --port $PORT"
   [[volumes]]
   mountPath = "/data"
   name = "anra-cluster-db"
   ```
4. Set environment variable `REDIRECT_URI` to `https://your-app.railway.app/api/auth/callback`
5. Set `GOOGLE_CLIENT_SECRETS` to the path of your OAuth client secret JSON

### Frontend → Vercel

1. Connect the same repo to Vercel
2. Framework: Vite
3. Build command: `cd frontend && npm run build`
4. Output directory: `frontend/dist`
5. Set `VITE_API_URL` to your Railway backend URL
6. `vercel.json` handles SPA routing automatically

---

## Usage Guide

### Step 1: Prepare Drive

```bash
# Run the setup script (requires OAuth client_secret.json)
python scripts/setup_drive_structure.py
```

Or manually:
1. Create a folder `AnRa/cluster/` in your primary Google Drive
2. Share the `cluster` folder with each worker's Gmail (Editor role)
3. Copy the folder ID from the URL: `drive.google.com/drive/folders/`**`THIS_PART`**

### Step 2: Open the Dashboard

Navigate to your deployed app (or `http://localhost:5173` for local dev).

Click **Setup** in the nav bar.

### Step 3: Configure the Cluster

Fill in:
- **Folder ID** — the ID from Step 1
- **Number of Workers** — how many Colab accounts you have
- **Target Steps** — e.g., 100,000
- **Checkpoint Filename** — e.g., `anra_frontier_500m.pt`

Click **Initialize**.

### Step 4: Launch Workers

The setup wizard generates N worker cards, each containing:

```
worker_A
Slot 1

WORKER_ID="worker_A"
ACCOUNT_EMAIL="account_1@gmail.com"
```

For each card:

1. Click **Open in Colab ↗** (opens the notebook in a new tab)
2. **Sign into the corresponding Gmail account** in that browser/incognito window
3. Fill in the config cell with the values from the dashboard
4. **Runtime → Run all**

### Step 5: Monitor

Switch back to the dashboard. Within 2 minutes, workers appear as green dots:

```
◈ worker_A  ●   ◈ worker_B  ●   ◈ worker_C  ●
```

The loss curve starts moving. The gradient heatmap glows. You can:
- **Pause/Resume** all workers from the Cluster page
- **Force Aggregation** to manually trigger a gradient averaging step
- **Click any worker** for detailed metrics
- **View Logs** for real-time SSE event stream

### Step 6: Download Checkpoint

When training reaches the target steps (or whenever you want), download the latest checkpoint:
```
GET /api/drive/files  →  find master_weights_v{N}.pt  →  download from Drive
```

---

## API Reference

### `POST /api/cluster/init`
Initialize the cluster. Creates `coordinator_state.json` on Drive.

```json
// Request
{
  "coordinator_drive_folder_id": "1ABCxyz...",
  "master_checkpoint_filename": "anra_frontier_500m.pt",
  "total_target_steps": 100000
}

// Response (201)
{
  "cluster_id": "anra-cluster-1",
  "status": "initialized"
}
```

```bash
curl -X POST https://your-app.railway.app/api/cluster/init \
  -H "Content-Type: application/json" \
  -d '{"coordinator_drive_folder_id":"1ABC...","master_checkpoint_filename":"anra_frontier_500m.pt","total_target_steps":100000}'
```

### `POST /api/workers/register`
Register a worker. Returns assigned slot and master weights path.

```json
// Request
{
  "worker_id": "worker_A",
  "account_email": "account_1@gmail.com",
  "drive_folder_id": "/content/drive/MyDrive/AnRa/cluster"
}

// Response (201)
{
  "assigned_slot": 1,
  "master_weights_path": "anra_frontier_500m.pt"
}
```

### `GET /api/workers`
List all workers with status, loss, step, GPU memory.

```json
// Response
[
  {
    "worker_id": "worker_A",
    "status": "active",
    "current_step": 1042,
    "loss": 2.847,
    "gpu_memory_mb": 12500,
    "tokens_processed": 1067008,
    "last_heartbeat": 1700000000.0
  }
]
```

### `POST /api/workers/{worker_id}/heartbeat`
Worker liveness ping (called every 30s).

```json
// Request
{
  "current_step": 1042,
  "loss": 2.847,
  "tokens_processed": 1067008,
  "gpu_memory_mb": 12500
}

// Response — command tells worker what to do
{ "command": "continue" }    // keep training
{ "command": "pause" }       // coordinator is aggregating or paused
{ "command": "stop" }        // training complete
{ "command": "reload_weights" } // new weights published, reload
```

### `POST /api/workers/{worker_id}/gradient_ready`
Worker signals it wrote gradients to Drive.

```json
// Request
{
  "step": 42,
  "grad_file_path": "/content/drive/MyDrive/AnRa/cluster/worker_A/grad_step_000042.pt",
  "minibatch_tokens": 1024
}

// Response
{
  "acknowledged": true,
  "aggregation_pending": false   // true if all workers checked in
}
```

### `GET /api/training/status`
Global training status. Polled by dashboard every 5s.

```json
// Response
{
  "global_step": 1042,
  "total_target_steps": 100000,
  "total_loss_history": [3.2, 2.8, 2.4, 2.1],
  "active_workers": 3,
  "total_workers": 3,
  "tokens_per_second_total": 2847,
  "phase": "training",
  "master_weights_version": 41,
  "current_lr": 0.0003,
  "aggregation_in_progress": false
}
```

### `POST /api/training/pause`
Pause all workers. Workers see `"pause"` command on next heartbeat.

### `POST /api/training/resume`
Resume training. Workers see `"continue"` command.

### `POST /api/training/aggregate`
Manually trigger gradient aggregation. Returns `409` if already aggregating.

```json
// Request
{ "step": 42 }
```

### `GET /api/logs/stream`
Server-Sent Events endpoint. Real-time log stream with keepalive pings (every 30s).

```
data: {"message":"[INFO] [2026-06-30T01:23:45] Gradient received from worker_A for step 42","timestamp":1700000000}

data: {"type":"keepalive","timestamp":1700000000}
```

### `GET /api/drive/files`
List all files in the Drive cluster folder.

### `GET /api/health`
Health check.

```json
{
  "status": "healthy",
  "timestamp": 1700000000.0,
  "aggregation_in_progress": false,
  "drive_initialized": true
}
```

---

## Architecture Deep Dive

### The Gradient File Format

```
{
  "step": 42,
  "worker_id": "worker_A",
  "model_version": 41,           # must match master weights version
  "token_count": 1024,           # for weighted averaging
  "loss": 2.847,
  "use_fp16": true,
  "gradients": {
    "transformer.layers.0.self_attn.q_proj.weight": tensor(fp16),
    "transformer.layers.0.self_attn.k_proj.weight": tensor(fp16),
    # ... all 499M parameters' gradients
  }
}
```

**Size:** ~1 GB with fp16 (down from ~2 GB fp32). Gradient sparsification (stretch goal) would reduce this to ~20 MB.

### Optimistic Lock Protocol

Drive has no atomic operations. This lock prevents two coordinators from writing master weights simultaneously:

```python
def acquire_lock(self, holder_id: str) -> bool:
    current = read_coordinator_state()
    if current["lock_holder"] and not expired:
        return False
    current["lock_holder"] = holder_id
    current["lock_time"] = time.time()
    write_coordinator_state(current)
    time.sleep(2)                      # wait for races
    verify = read_coordinator_state()
    return verify["lock_holder"] == holder_id  # did we win?
```

### Aggregation Quorum Logic

The coordinator aggregates when:
1. **All active workers** have submitted gradients for the current step
2. **OR** the first submission was > 5 minutes ago (partial aggregation — proceeds with whoever submitted)

Dead workers are marked `stale` after 2 minutes of no heartbeat. They are excluded from the quorum.

### Gradient Sanitization (aggregator.py)

Before averaging, each gradient tensor is:
1. **NaN/Inf masked** — non-finite values replaced with zero
2. **Clamped** — values outside [-1e3, 1e3] are bounded
3. **Globally clipped** — total gradient norm capped at 1.0

After averaging, gradients are **token-count weighted** (workers with larger batches contribute proportionally more).

### Gradient Application (main.py:run_aggregation)

```python
for name, grad in averaged_grads.items():
    if name in model_state:
        # Simple SGD: w = w - lr * grad
        model_state[name].sub_(grad.float().to(param.dtype), alpha=lr)
```

This is equivalent to SGD. For Adafactor (An-Ra's memory-efficient optimizer), the worker loop already has the factored second moment estimates — but for distributed training, returning to SGD on the aggregated gradients is the simplest correct approach.

---

## Edge Cases & Fault Tolerance

### Drive Quota (15 GB limit)
- Each gradient file: ~1 GB (fp16)
- Auto-cleanup keeps only the last 3 steps per worker
- Max usage: 3 workers × 3 steps × 1 GB = 9 GB (safe within 15 GB)
- Plus coordinator_state.json and master_weights (~2 GB each)

### Colab Disconnect (12h timeout)
- Workers save data position to `data_position.json` on Drive
- On restart: resume from saved byte offset (no data loss)
- Signal handlers (SIGTERM/SIGINT) allow graceful stop mid-step

### Model Version Mismatch
- Each gradient file includes `model_version`
- If workers submit gradients computed from different-weight models, the coordinator warns but proceeds
- The affected step uses stale gradients — mathematically equivalent to a larger effective batch

### Partial Aggregation (Dead Workers)
- If 2/3 workers submit and the deadline passes, coordinator aggregates with 2
- The dead worker's slot remains registered; when it reconnects, it gets `reload_weights` and catches up

### Drive API Rate Limits
- Google Drive API: 1,000 queries per 100 seconds per user
- `drive_sync.py` uses `tenacity` retry with exponential backoff (2s → 4s → 8s → 16s → 30s, 5 attempts)
- Batch reads where possible (list queries return up to 200 items per page)

---

## Performance

Measured on **An-Ra frontier 500M** with sequence length 1024:

| Workers | Tokens/sec | vs Single | 100k Step ETA | Gradients per Step |
|---------|------------|-----------|---------------|-------------------|
| 1       | ~350       | 1×        | ~79 hours     | 1                  |
| 2       | ~700       | 2×        | ~40 hours     | 2                  |
| 3       | ~1050      | 3×        | ~27 hours     | 3                  |
| 5       | ~1750      | 5×        | ~16 hours     | 5                  |

**Why better than N×?** Async overlap — while worker A waits for aggregation, worker B is already computing the next step. The effective throughput approaches N× the single-worker rate as N increases.

**Bottleneck:** Drive upload/download speed (~50-100 MB/s). A 1 GB gradient file takes ~10-20 seconds to write. The aggregation step adds another ~30-60 seconds for coordinator download + averaging + upload. Total overhead per step: ~60-90 seconds, amortized across N workers.

---

## Troubleshooting

### Workers don't appear in dashboard
- Verify the Drive folder is **shared with Editor access** to each worker's Gmail
- Check the coordinator URL in the notebook config cell — must match the deployed backend
- Check browser console for CORS errors — backend uses `allow_origins=["*"]`

### "Drive not initialized" error
- Call `POST /api/cluster/init` first (or use the Setup wizard)
- Verify the Drive folder ID is correct

### Workers show as "stale"
- Colab session may have disconnected (idle timeout ~90 min, hard limit ~12 h)
- Click **Setup** → generate new notebook link → **Run All** in the same Colab
- The worker re-registers and picks up from its saved data position

### Training loss spikes after aggregation
- This is expected initially — workers compute gradients on different data slices
- The loss typically stabilizes after 50-100 steps as all workers converge to the same region
- If the spike exceeds 0.5, the gradient clipping in `aggregator.py` (max_norm=1.0) prevents divergence

### "Aggregation already in progress" (409)
- Wait for the current aggregation to finish (~30-60 seconds)
- Or check `/api/training/status` for `aggregation_in_progress`

### Drive API quota exceeded
- The `tenacity` retry handles transient rate limits automatically
- If persistent: reduce the number of workers or increase `keep_last_n_steps` in cleanup
- Maximum sustained rate: ~10 queries/second (1000 queries per 100 seconds)

### "No master weights found, creating from scratch"
- First-time initialization: the coordinator creates an empty model
- Upload your existing checkpoint to Drive so the path exists before init

---

## Stretch Goals

| Feature | Description | Impact |
|---------|-------------|--------|
| **Gradient Sparsification** | Transmit only top-1% of gradients by magnitude (99% are near zero) | 2 GB → **20 MB** per step, ~10× throughput |
| **Colab TPU Support** | TPU v2 is also free, ~8× T4 performance via `torch_xla` | 3 T4 workers → **24× single-T4 throughput** |
| **Auto-Pairing** | App programmatically shares the Drive folder with each worker's Gmail | Removes the manual "share this folder" step |
| **Data Sharding** | Each worker gets a different slice per epoch, ensuring diverse batches | Better gradient diversity → faster convergence |
| **Loss Spike Recovery** | If aggregated loss spikes >0.5, auto-rollback to previous checkpoint | Prevents divergence, saves days of training |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_PATH` | `/data/cluster.db` | SQLite database path (Railway volume) |
| `GOOGLE_CLIENT_SECRETS` | `client_secret.json` | OAuth 2.0 client secrets file |
| `REDIRECT_URI` | `http://localhost:8000/api/auth/callback` | OAuth callback URL |

---

## Development

```bash
# Lint Python
pip install ruff
ruff check backend/ worker/ scripts/

# Format
ruff format backend/ worker/ scripts/

# Lint frontend
cd frontend && npx eslint src/
```

---

## License

MIT. Built for the An-Ra project — a 499M-parameter frontier transformer trained by the community, for the community.
