# AI Customer Support Agent — Hiver SDE Intern Take-Home

An end-to-end (but intentionally small) system that takes an incoming customer
message and:

1. Classifies it into one of 7 intents.
2. Retrieves the most similar historical, resolved conversations (TF-IDF + cosine similarity).
3. Drafts a reply grounded in those historical resolutions.
4. Decides `AUTO_HANDLE` vs `ESCALATE`, with a confidence score and a human-readable reason.
5. Shows the retrieved evidence in the UI so the decision is auditable, not a black box.

Everything runs **fully offline with no API key** using deterministic fallbacks. An LLM
(Anthropic Claude) can optionally be plugged in via `.env` for reply generation and the
LLM-as-judge evaluation script — see [Environment variables](#environment-variables).

> **On the dataset**: this project is built around the schema of the Kaggle
> ["Customer Support on Twitter"](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
> dataset (`twcs.csv`), but that file is **not included** in this ZIP (it's large and
> Kaggle-licensed). A small, realistic **demo dataset** (`backend/data/demo_conversations.csv`,
> 112 rows) ships instead so the app runs immediately. See
> [Using the real Kaggle dataset](#using-the-real-kaggle-dataset-optional) to plug in the real
> data later.

---

## Contents

- [Project structure](#project-structure)
- [Quick start (Windows)](#quick-start-windows)
- [Environment variables](#environment-variables)
- [Backend](#backend)
- [Frontend](#frontend)
- [Evaluation](#evaluation)
- [Using the real Kaggle dataset (optional)](#using-the-real-kaggle-dataset-optional)
- [Reproducing everything in under 15 minutes](#reproducing-everything-in-under-15-minutes)
- [Troubleshooting](#troubleshooting)

## Project structure

```
hiver-sde-intern/
├── backend/            Flask REST API (classification, retrieval, reply, escalation)
├── frontend/            React + Vite dashboard
├── evaluation/          Golden set, metrics, baselines, LLM-as-judge
├── README.md            You are here
├── REPORT.md            Problem framing, results, failure modes, limitations
├── DECISION_LOG.md      Engineering decisions and why they were made
└── .gitignore
```

## Quick start (Windows)

These steps assume **PowerShell** and that Python 3.10+ and Node.js 18+ are installed.
(Check with `python --version` and `node --version` in PowerShell.)

### 1. Unzip the project

Unzip `hiver-sde-intern-complete.zip` anywhere, e.g. `C:\Users\you\hiver-sde-intern`, and
open PowerShell in that folder.

### 2. Backend setup

```powershell
cd backend
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
python app.py
```

You should see:

```
 * Running on http://127.0.0.1:5001
```

Leave this window open. Backend is now serving the API on port `5001`.

> If PowerShell blocks `Activate.ps1` with an execution-policy error, run:
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that same window, then
> retry.

### 3. Frontend setup (in a **new** PowerShell window)

```powershell
cd frontend
copy .env.example .env
npm install
npm run dev
```

Open the URL it prints (usually `http://localhost:5173`) in your browser. Type a customer
message and press **Analyze**.

### 4. (Optional) Run the automated tests

```powershell
cd backend
venv\Scripts\Activate.ps1
pytest
```

## Environment variables

### `backend/.env` (copy from `backend/.env.example`)

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `none` | `none` = fully offline deterministic replies. `anthropic` = use Claude for reply generation. |
| `ANTHROPIC_API_KEY` | (blank) | Required only if `LLM_PROVIDER=anthropic`. Never commit a real key. |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-5` | Model name, only used if the LLM path is enabled. |
| `RETRIEVAL_TOP_K` | `3` | How many historical conversations to retrieve as evidence. |
| `INTENT_CONFIDENCE_ESCALATION_THRESHOLD` | `0.45` | Below this classifier confidence, always escalate. |
| `RETRIEVAL_SIMILARITY_ESCALATION_THRESHOLD` | `0.15` | Below this best-match similarity, always escalate. |
| `ALWAYS_ESCALATE_INTENTS` | `billing_and_payment` | Comma-separated intents that always escalate regardless of confidence. |
| `FLASK_PORT` | `5001` | Backend port. |
| `CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Allowed frontend origins. |

The app **never hardcodes an API key** anywhere in the codebase — it is only ever read
from the environment.

### `frontend/.env` (copy from `frontend/.env.example`)

| Variable | Default | Meaning |
|---|---|---|
| `VITE_API_BASE_URL` | `http://localhost:5001` | Where the frontend sends API requests. |

## Backend

Flask app in `backend/app.py`. Endpoints:

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Service status, corpus size, which classifier/retrieval backend is active. |
| GET | `/api/brands` | Distinct brand handles in the loaded corpus. |
| GET | `/api/intents` | The 7 supported intents. |
| POST | `/api/analyze` | Body: `{"message": "..."}`. Returns intent, confidence, draft reply, decision, reason, evidence. |

Modules:

- `classifier.py` — TF-IDF + Logistic Regression, trained at startup from the demo corpus,
  with a zero-dependency keyword-rule fallback (works even without scikit-learn installed).
- `retrieval.py` — TF-IDF + cosine similarity, with a pure-Python fallback.
- `reply_generator.py` — deterministic template-based reply (default) or LLM-generated
  reply (if configured), always grounded in the retrieved evidence.
- `escalation.py` — pure, auditable rule logic for `AUTO_HANDLE` vs `ESCALATE`.
- `config.py` — all configuration, read from environment variables.

## Frontend

React + Vite dashboard in `frontend/`. Single page: message textbox, sample messages you can
click to try, an Analyze button, and a results panel showing the intent, confidence,
decision banner (auto-handle / escalate + reason), the draft reply, and the retrieved
evidence with similarity scores. Handles loading and error states (e.g. backend not
running, empty message, network failure).

## Evaluation

From the `evaluation/` folder (backend dependencies must be installed first, since the
evaluation script imports the real classifier/retrieval/escalation code — it does not
reimplement or fake them):

```powershell
cd evaluation
..\backend\venv\Scripts\Activate.ps1
python evaluate.py
```

This prints, using the actual current code and the 196-row golden set:

- Intent accuracy + macro F1 for the main system **and** two baselines
  (`baselines.py`: majority-class trivial baseline, TF-IDF+nearest-centroid simple baseline).
- Escalation precision / recall / F1 against human-judged expected labels.
- Retrieval Recall@K (proxy metric — see the script's docstring for exactly what it measures
  and does not measure).
- A reply-quality *heuristic* proxy (word overlap / length) — **not** the LLM-judge rubric.

Real numbers from the last run of this script are in `REPORT.md`, clearly not fabricated.

To also run the **LLM-as-judge** rubric (grounded / relevant / tone / actionable, 1-5 each),
set `ANTHROPIC_API_KEY` in your environment, then:

```powershell
python evaluate.py --predictions-out predictions.csv
python judge.py --predictions predictions.csv --out judge_results.csv
```

Without an API key, `judge.py` writes `"NOT RUN"` for every row rather than making anything up.
See `REPORT.md` → "What was not built" for the human-vs-LLM agreement process, which has not
been executed for this submission (there are no human labels yet).

`golden_set.csv` was generated from templates by `generate_golden_set.py` (kept for
transparency/reproducibility — re-running it with the same seed reproduces the same file).

## Using the real Kaggle dataset (optional)

1. Download `twcs.csv` from Kaggle's
   [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
   dataset and place it anywhere on disk (it is **not** included in this project).
2. Preprocess it into the pair schema this project uses:
   ```powershell
   cd backend
   python scripts\preprocess.py --input C:\path\to\twcs.csv --output data\full_processed.csv --limit 20000
   ```
3. Create a small, stratified demo subset from the real data (replaces the synthetic one):
   ```powershell
   python scripts\create_subset.py --source data\full_processed.csv
   ```
   This overwrites `backend/data/demo_conversations.csv`. Restart `app.py` to pick it up.

## Reproducing everything in under 15 minutes

```powershell
# 1. Backend (≈3 min incl. installs)
cd backend
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
pytest                          # ≈1 second, 14 tests
Start-Process python app.py     # leave running

# 2. Frontend (≈3 min incl. installs), new window
cd ..\frontend
copy .env.example .env
npm install
npm run dev                     # leave running, open printed URL

# 3. Evaluation (≈10 seconds), new window
cd ..\evaluation
..\backend\venv\Scripts\Activate.ps1
python evaluate.py
```

Total: well under 15 minutes on a normal machine (most of the time is `pip install` /
`npm install`).

## Troubleshooting

- **`Activate.ps1 cannot be loaded because running scripts is disabled`**: run
  `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that PowerShell window.
- **Frontend shows "backend unreachable"**: confirm `python app.py` is still running and
  `frontend/.env`'s `VITE_API_BASE_URL` matches the port Flask printed.
- **`ModuleNotFoundError` for scikit-learn**: the app still works (falls back to the
  zero-dependency keyword/cosine implementations) but re-run
  `pip install -r requirements.txt` inside the activated venv to get the full ML path.
- **Port already in use**: change `FLASK_PORT` in `backend/.env` (and `VITE_API_BASE_URL`
  in `frontend/.env` to match).
