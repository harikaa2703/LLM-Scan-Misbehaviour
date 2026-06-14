# LLMSCAN-MISBEHAVIOUR

**AI model safety scanner for evaluating LLM responses on safety, reliability, and trustworthiness before deployment in real applications.**

LLMSCAN lets users submit prompts, evaluate responses across multiple open-source LLMs, and inspect safety scores, behavioral indicators, and visual analysis.

## System Architecture

```
User Prompt
    ↓
React Frontend (Dashboard UI)
    ↓
FastAPI Backend
    ↓
Selected LLM (TinyLlama / GPT-2 / GPT-Neo / Mistral)
    ↓
Feature Engine (65-D Vector)
    ↓
MLP Classifier
    ↓
Safety Scores & Risk Analysis
    ↓
Explainability (Heat Maps, Causal Maps)
    ↓
SQLite Storage
```

## Supported Models

| Model | Type | Purpose |
|---|---|---|
| TinyLlama | Lightweight LLM | Fast safety evaluation |
| GPT-2 | Transformer LLM | Baseline comparison |
| GPT-Neo 125M | Open Source LLM | Behavioral benchmarking |
| Mistral | Advanced LLM | Robust safety assessment |

## Features

- Scan prompts against TinyLLaMA, GPT-2, GPT-Neo, and Mistral
- Generate a custom **65-dimensional feature vector** representing model behavior and safety characteristics
- Detect toxicity, bias, jailbreak behavior, hallucination, harmful intent, deception, vulnerability exploitation, confidence anomalies, entropy-based uncertainty, and general misbehavior
- Classify responses using a custom **MLP safety classifier**
- Display risk summaries and detailed score breakdowns
- Generate visual analytics: causal analysis, token confidence, entropy analysis, behavioral consistency plots, heatmaps, explainability visualizations
- Compare safety performance across multiple foundation models
- Save scan history locally via SQLite
- React dashboard for running scans, comparisons, and viewing history

## Tech Stack

**Backend**
- Python 3.12, FastAPI, Uvicorn, PyTorch, Hugging Face Transformers, Sentence Transformers, scikit-learn, SQLite

**Frontend**
- React, TypeScript, Vite, Tailwind CSS, Axios, Recharts, Framer Motion

**Machine Learning**
- Custom MLP Classifier
- 65-Dimensional Feature Engineering Pipeline
- Explainable AI Components
- Safety Evaluation Modules

## Project Structure

```
llmscan/
├── main.py
├── requirements.txt
├── scans.db
├── scanners/
├── frontend/
├── tests/
├── scripts/
├── Dockerfile
├── docker-compose.yml
├── run.bat
└── run.sh
```

## Requirements

- Python 3.12 recommended (3.14 not recommended — some ML dependencies may not support it cleanly yet)
- Node.js 18+ and npm
- Internet connection for first-time model downloads
- Optional: CUDA-capable GPU for faster inference

## Setup

**Backend** (from project root)
```bash
py -3.12 -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```
Runs at `http://localhost:5000` · Docs at `http://localhost:5000/docs`

**Frontend** (second terminal)
```bash
cd frontend
npm install
npm run dev
```
Runs at `http://localhost:5173`

## Safety Evaluation Workflow

1. User submits a prompt and selects a model
2. Model generates a response
3. 65-D feature extraction pipeline analyzes the output
4. Safety evaluators calculate toxicity, hallucination, jailbreak, deception, confidence, entropy, and vulnerability indicators
5. Features passed into custom MLP classifier
6. Risk scores and safety labels generated
7. Explainability modules create visual insights and causal analysis
8. Results stored in SQLite and displayed in dashboard

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Checks whether the backend is running |
| POST | `/scan` | Runs a safety scan for a prompt and selected model |
| GET | `/progress` | Returns current scan progress |
| GET | `/models` | Returns available model options |
| POST | `/causal-map` | Generates causal/layer analysis |
| GET | `/metrics` | Returns evaluation metric information |
| GET | `/history` | Returns saved scan history |
| POST | `/load` | Loads a previous scan result |
| DELETE | `/history/{item_id}` | Deletes a saved scan |

**Supported model values:** `tinyllama`, `gpt2`, `gptneo`, `mistral`

**Example scan request:**
```json
{
  "prompt": "Explain how to stay safe online",
  "model": "tinyllama"
}
```

## How Results Are Stored

LLMSCAN uses a local SQLite database (`scans.db`) storing prompts, selected models, safety classifications, timestamps, extracted features, and full JSON scan results in a scan history table. An in-memory cache avoids recomputing previously evaluated prompt-model combinations.

To exclude scan history from GitHub, add to `.gitignore`:
```
scans.db
```

## Notes

- First scan can take several minutes (models may need to download)
- Model files are cached locally after first download
- Some Hugging Face models may require login/access approval
- Backend uses CPU for safety evaluators and embeddings to reduce GPU memory requirements
- If locally stored MLP `.pkl` models fail to load due to compatibility issues, fallback safety evaluators are used automatically

## Common Commands

| Action | Command |
|---|---|
| Run backend | `.\venv\Scripts\activate` → `python main.py` |
| Run frontend | `cd frontend` → `npm run dev` |
| Build frontend | `cd frontend` → `npm run build` |
| Check Git status | `git status` |

## Purpose

LLMSCAN supports safer AI development by combining feature engineering, machine learning, explainability techniques, and behavioral analysis to identify risks and improve transparency in modern AI systems — going beyond raw response display to surface trustworthiness insights.