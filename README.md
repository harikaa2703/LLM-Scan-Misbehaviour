

LLMSCAN-MISBEHAVIOUR

LLMSCAN is an AI model safety scanner that helps evaluate whether an LLM response is safe, reliable, and trustworthy before it is used in a real application.

Large Language Models can generate toxic, biased, hallucinated, misleading, deceptive, or jailbreak-prone responses. LLMSCAN enables users to submit prompts, evaluate responses from multiple open-source language models, and inspect safety scores, behavioral indicators, and visual analysis generated from the model output.
System Architecture

                    User Prompt
                         │
                         ▼
                ┌────────────────┐
                │ React Frontend │
                │ Dashboard UI   │
                └────────────────┘
                         │
                         ▼
                ┌────────────────┐
                │ FastAPI Backend│
                └────────────────┘
                         │
                         ▼
                ┌────────────────┐
                │ Selected LLM   │
                │ TinyLlama      │
                │ GPT-2          │
                │ GPT-Neo        │
                │ Mistral        │
                └────────────────┘
                         │
                         ▼
                ┌────────────────┐
                │ Feature Engine │
                │ 65-D Vector    │
                └────────────────┘
                         │
                         ▼
                ┌────────────────┐
                │ MLP Classifier │
                └────────────────┘
                         │
                         ▼
                ┌────────────────┐
                │ Safety Scores  │
                │ Risk Analysis  │
                └────────────────┘
                         │
                         ▼
                ┌────────────────┐
                │ Explainability │
                │ Heat Maps      │
                │ Causal Maps    │
                └────────────────┘
                         │
                         ▼
                ┌────────────────┐
                │ SQLite Storage │
                └────────────────┘

Supported Models

Model

Type

Purpose

TinyLlama

Lightweight LLM

Fast safety evaluation

GPT-2

Transformer LLM

Baseline comparison

GPT-Neo 125M

Open Source LLM

Behavioral benchmarking

Mistral

Advanced LLM

Robust safety assessment
Features

- Scan prompts against supported LLMs: TinyLLaMA, GPT-2, GPT-Neo, and Mistral.
- Generate a custom 65-dimensional feature vector representing model behavior and safety characteristics.
- Detect safety risks such as toxicity, bias, jailbreak behavior, hallucination, harmful intent, deception, vulnerability exploitation, confidence anomalies, entropy-based uncertainty, and general misbehavior.
- Classify model responses using a custom Multi-Layer Perceptron (MLP) safety classifier.
- Show model responses with risk summaries and detailed score breakdowns.
- Generate visual analytics including causal analysis, token confidence, entropy analysis, behavioral consistency plots, heatmaps, and explainability visualizations.
- Compare safety performance across multiple foundation models.
- Save previous prompts and scan results locally using SQLite.
- Provide a React dashboard for running scans, comparing models, and viewing evaluation history.

Tech Stack

Backend

- Python 3.12
- FastAPI
- Uvicorn
- PyTorch
- Hugging Face Transformers
- Sentence Transformers
- scikit-learn
- SQLite

Frontend

- React
- TypeScript
- Vite
- Tailwind CSS
- Axios
- Recharts
- Framer Motion

Machine Learning

- Custom MLP Classifier
- 65-Dimensional Feature Engineering Pipeline
- Explainable AI Components
- Safety Evaluation Modules

Project Structure

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

Requirements

- Python 3.12 recommended
- Node.js 18+ and npm
- Internet connection for first-time model downloads
- Optional: CUDA-capable GPU for faster model inference

Python 3.14 is not recommended for this project because some ML dependencies may not support it cleanly yet.

Backend Setup

From the project root:

py -3.12 -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py

The backend runs at:

http://localhost:5000

API documentation is available at:

http://localhost:5000/docs

Frontend Setup

Open a second terminal:

cd frontend
npm install
npm run dev

The frontend runs at:

http://localhost:5173

Safety Evaluation Workflow

1. User submits a prompt and selects a language model.
2. The selected model generates a response.
3. A 65-dimensional feature extraction pipeline analyzes the generated output.
4. Safety evaluators calculate toxicity, hallucination, jailbreak, deception, confidence, entropy, and vulnerability indicators.
5. Features are passed into a custom MLP classifier.
6. Risk scores and safety labels are generated.
7. Explainability modules create visual insights and causal analysis.
8. Results are stored in SQLite and displayed in the dashboard.

API Endpoints

Method| Endpoint| Description
GET| /health| Checks whether the backend is running
POST| /scan| Runs a safety scan for a prompt and selected model
GET| /progress| Returns current scan progress
GET| /models| Returns available model options
POST| /causal-map| Generates causal/layer analysis
GET| /metrics| Returns evaluation metric information
GET| /history| Returns saved scan history
POST| /load| Loads a previous scan result
DELETE| /history/{item_id}| Deletes a saved scan

Supported Models

- tinyllama
- gpt2
- gptneo
- mistral

Example scan request:

{
  "prompt": "Explain how to stay safe online",
  "model": "tinyllama"
}

How Results Are Stored

LLMSCAN uses a local SQLite database file:

scans.db

The backend stores prompts, selected models, safety classifications, timestamps, extracted features, and full JSON scan results in a scan history table. An in-memory cache is also maintained to avoid recomputing previously evaluated prompt-model combinations.

If you do not want to upload scan history to GitHub, add the following to ".gitignore":

scans.db

Notes

- The first scan can take several minutes because language models and evaluation models may need to download.
- Model files are cached locally after the first download.
- Some Hugging Face models may require login or access approval.
- The backend is configured to use CPU for safety evaluators and embeddings to reduce GPU memory requirements.
- If locally stored MLP ".pkl" models fail to load due to compatibility issues, fallback safety evaluators are used automatically.

Common Commands

Run backend:

.\venv\Scripts\activate
python main.py

Run frontend:

cd frontend
npm run dev

Build frontend:

cd frontend
npm run build

Check Git status:

git status

Purpose

LLMSCAN is designed to support safer AI development by providing a practical framework for evaluating model trustworthiness. Instead of only displaying generated responses, the platform combines feature engineering, machine learning, explainability techniques, and behavioral analysis to identify risks and improve transparency in modern AI systems.This version looks like an evolution of the original README rather than a completely different document, while showcasing the parts that make your project stronger.

