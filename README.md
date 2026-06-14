# 🔍 LLMSCAN-MISBEHAVIOUR

> Catch toxic, biased, hallucinated, or jailbreak-prone LLM outputs before they ship.

LLMSCAN is a safety evaluation toolkit for open-source language models. Submit a prompt, pick a model, and get back risk scores, behavioral breakdowns, and visual explainability — all from a local dashboard.

---

## 📋 Table of Contents

- [What It Does](#-what-it-does)
- [Quick Start](#-quick-start)
- [How It Works](#-how-it-works)
- [Supported Models](#-supported-models)
- [API](#-api)
- [Tech Stack](#-tech-stack)
- [Caveats](#-caveats)

---

## 🧠 What It Does

LLMSCAN runs prompts through one or more LLMs and analyzes the responses for:

- **Toxicity** — harmful or offensive language
- **Bias** — skewed or unfair outputs
- **Jailbreak susceptibility** — does the model break its own guardrails
- **Hallucination** — fabricated or unsupported claims
- **Deception / harmful intent** — misleading or manipulative responses
- **Confidence & entropy anomalies** — signals of uncertainty or instability
- **General misbehavior** — anything else flagged as risky

Each response is converted into a **65-dimensional feature vector** and run through a custom **MLP classifier**, producing risk scores plus heatmaps, causal maps, and consistency plots.

---

## ⚡ Quick Start

**1. Backend**
```bash
py -3.12 -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python main.py
```
→ `http://localhost:5000` (docs at `/docs`)

**2. Frontend**
```bash
cd frontend
npm install
npm run dev
```
→ `http://localhost:5173`

**Requirements:** Python 3.12 (not 3.14), Node.js 18+, internet for first-run model downloads, optional CUDA GPU.

---

## 🔄 How It Works

```
Prompt + Model Selection
    ↓
LLM generates response  (TinyLlama / GPT-2 / GPT-Neo / Mistral)
    ↓
65-D feature extraction
    ↓
MLP safety classifier
    ↓
Risk scores + explainability visuals
    ↓
Saved to SQLite, shown on dashboard
```

---

## 🤖 Supported Models

| Model | Type | Best For |
|---|---|---|
| TinyLlama | Lightweight | Fast iteration |
| GPT-2 | Transformer | Baseline comparison |
| GPT-Neo 125M | Open source | Behavioral benchmarking |
| Mistral | Advanced | Robust safety checks |

---

## 📡 API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/models` | Available models |
| POST | `/scan` | Run a safety scan |
| GET | `/progress` | Scan progress |
| POST | `/causal-map` | Causal/layer analysis |
| GET | `/metrics` | Metric definitions |
| GET | `/history` | Past scans |
| POST | `/load` | Reload a past scan |
| DELETE | `/history/{item_id}` | Delete a scan |

**Example:**
```json
{ "prompt": "Explain how to stay safe online", "model": "tinyllama" }
```
Model values: `tinyllama`, `gpt2`, `gptneo`, `mistral`

---

## 🛠️ Tech Stack

**Backend:** Python 3.12 · FastAPI · PyTorch · Hugging Face Transformers · Sentence Transformers · scikit-learn · SQLite
**Frontend:** React · TypeScript · Vite · Tailwind CSS · Recharts · Framer Motion
**ML:** Custom MLP classifier · 65-D feature pipeline · explainability modules

Scan history is stored locally in `scans.db`. Add it to `.gitignore` if you don't want it tracked.

---

## ⚠️ Caveats

- First scan is slow (models download on first use, then cache locally)
- Some HF models may need login/access approval
- Runs on CPU by default to keep GPU memory free
- Corrupted/incompatible `.pkl` classifiers fall back to default evaluators automatically