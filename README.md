# LLMSCAN - AI Model Safety Scanner

A modern, sleek web application for scanning AI models for safety issues. Built with React, TypeScript, Tailwind CSS, and Flask.

## 📋 Project Structure

```
Domain/
├── backend.py              # Flask API server
├── requirements.txt        # Python dependencies
├── tinyllama_scanner.py    # AI model scanning logic
├── gptneo_scanner.py       # GPT-Neo model scanner
├── gpt2_scanner.py         # GPT-2 model scanner
└── frontend/
    ├── src/
    │   ├── App.tsx         # Main React application
    │   ├── main.tsx        # Entry point
    │   ├── index.css       # Global styles
    │   └── components/
    │       ├── Sidebar.tsx
    │       ├── PromptInput.tsx
    │       ├── ModelSelector.tsx
    │       ├── ResponseCard.tsx
    │       ├── MetricCard.tsx
    │       ├── ScoreBar.tsx
    │       ├── LayerChart.tsx
    │       └── PlotContainer.tsx
    ├── index.html
    ├── package.json
    ├── tsconfig.json
    ├── vite.config.ts
    ├── tailwind.config.js
    └── postcss.config.js
```

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ and npm
- Python 3.8+
- CUDA (optional, for GPU support)

### 1. Setup Backend

```bash
# Navigate to Domain directory
cd Domain

# Create and activate virtual environment
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the backend server
python backend.py
```

The backend will start on `http://localhost:5000`

### 2. Setup Frontend

```bash
# In a new terminal, navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will be available at `http://localhost:5173`

## 📡 API Endpoints

### POST /scan
Submits a prompt for scanning.

**Request:**
```json
{
  "prompt": "Your prompt here",
  "model": "tinyllama" | "gpt2" | "gptneo"
}
```

**Response:**
```json
{
  "success": true,
  "model": "tinyllama",
  "data": {
    "model": "TinyLlama-1.1B-Chat-v1.0",
    "prompt": "...",
    "baseline_response": "...",
    "baseline_scores": {
      "toxicity_score": 0.0,
      "bias_score": 0.0,
      "harmful_intent_score": 0.0,
      "jailbreak_score": 0.0,
      "hallucination_score": 0.0,
      "misbehaviour_score": 0.0,
      "overall_risk_score": 0.0
    },
    "layer_analysis": {
      "influence_scores": [...]
    },
    "hallucination_details": {...},
    "token_confidence_features": {...}
  }
}
```

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "model": "TinyLlama scanner ready"
}
```

### GET /models
Get available models.

## 🎨 UI Features

- **Dark Theme**: ChatGPT-style dark interface
- **Real-time Analytics**: Displays all metrics immediately after scan
- **Interactive Visualizations**: Layer influence, token confidence, consistency checks
- **Responsive Design**: Works on all screen sizes
- **Smooth Animations**: Framer Motion for polished transitions
- **Glassmorphism**: Modern UI with subtle gradients and shadows

## 📊 Metrics Displayed

- **Hallucination Score**: Detects when model generates false information
- **Toxicity Score**: Checks for harmful or toxic content
- **Bias Score**: Identifies biased responses
- **Jailbreak Detection**: Detects prompt injection attempts
- **Entropy**: Measures model's confidence
- **Layer Influence**: Shows which transformer layers are most sensitive to noise
- **Token Confidence**: Per-token probability analysis
- **Consistency Check**: Tests output stability across multiple runs

## 🛠️ Tech Stack

**Frontend:**
- React 18
- TypeScript
- Tailwind CSS
- Framer Motion (animations)
- Recharts (data visualizations)
- Axios (HTTP client)
- Vite (build tool)

**Backend:**
- Flask
- Flask-CORS
- PyTorch
- Hugging Face Transformers
- Sentence Transformers

## 🔧 Development

### Build Frontend
```bash
cd frontend
npm run build
```

### Lint Frontend
```bash
cd frontend
npm run lint
```

### Run Frontend in Production
```bash
cd frontend
npm run preview
```

## 📝 Notes

- The first model download will take several minutes (2.2GB for TinyLlama)
- Subsequent runs will be faster as models are cached
- For GPU support, ensure CUDA and appropriate PyTorch version are installed
- The backend processes one scan at a time

## 🔒 Safety Features

The scanner evaluates:
1. **Toxicity**: Using toxic-bert model
2. **Hate Speech**: Using RoBERTa hate-speech model
3. **Prompt Injection**: Using ProtectAI deberta model
4. **Jailbreak Attempts**: Pattern matching and model-based detection
5. **Hallucinations**: SelfCheckGPT methodology
6. **Layer Sensitivity**: Noise injection to identify critical layers

## 🐛 Troubleshooting

**Backend won't start:**
- Ensure Python 3.8+ is installed
- Check that port 5000 is available
- Verify all dependencies are installed: `pip install -r requirements.txt`

**Frontend won't connect to backend:**
- Ensure backend is running on `http://localhost:5000`
- Check browser console for CORS errors
- Verify firewall isn't blocking localhost connections

**Models won't download:**
- Set HuggingFace token: `huggingface-cli login`
- Check internet connection
- Try increasing timeout

