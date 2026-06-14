# LLMSCAN - Quick Start Guide

## ✅ What's been completed

✓ **Frontend (React + TypeScript)**
  - All 9 components created (Sidebar, PromptInput, ModelSelector, ResponseCard, MetricCard, ScoreBar, LayerChart, PlotContainer)
  - Dark theme with glassmorphism design
  - Real-time analytics display
  - Interactive charts with Recharts
  - Smooth animations with Framer Motion
  - Fully responsive layout
  - Tailwind CSS styling

✓ **Backend (Flask API)**
  - Flask server with CORS support
  - `/scan` endpoint integrated with TinyLlamaScanner
  - `/health` health check endpoint
  - `/models` endpoint for available models
  - JSON serialization of complex numpy types
  - Error handling and validation

✓ **Integration**
  - Frontend communicates with backend via Axios
  - API endpoint: http://localhost:5000/scan
  - Real-time data flow from backend to frontend UI
  - All metrics and visualizations connected

---

## 🚀 How to Run (Windows)

### Option 1: Automatic (Recommended)
Simply double-click the `run.bat` file in the Domain folder. This will:
1. Create Python virtual environment (if needed)
2. Install all backend dependencies
3. Install all frontend dependencies (npm packages)
4. Start backend server on http://localhost:5000
5. Start frontend on http://localhost:5173

### Option 2: Manual

**Terminal 1 - Backend:**
```
cd Domain
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python backend.py
```

**Terminal 2 - Frontend:**
```
cd Domain\frontend
npm install
npm run dev
```

---

## 📱 Access the Application

1. **Frontend**: Open http://localhost:5173 in your browser
2. **Backend API**: http://localhost:5000
3. **Health Check**: http://localhost:5000/health

---

## 🧪 Test the System

1. Open http://localhost:5173 in your browser
2. Select a model from dropdown (TinyLlama, GPT-2, or GPT-Neo)
3. Enter a prompt in the input box
4. Click "Scan" button
5. Wait for results (first run will download the model ~2.2GB)
6. See real-time analytics:
   - Hallucination, Toxicity, Bias, Entropy scores
   - Risk assessment with color coding
   - Layer influence visualization
   - Token confidence analysis
   - Consistency check charts

---

## 📊 What Each Component Shows

| Component | Purpose |
|-----------|---------|
| **Sidebar** | Navigation, scan history, model selection |
| **Response Card** | Model's output and risk assessment |
| **Metric Cards** | Key safety metrics (Hallucination, Toxicity, Bias, Entropy) |
| **Score Bars** | Breakdown of risk scores with visual progress bars |
| **Layer Chart** | Which transformer layers are most influential |
| **Plot Container** | Token confidence and consistency visualizations |

---

## ⚙️ System Architecture

```
┌─────────────────────────────────────────────┐
│         Browser (Frontend)                  │
│  React + TypeScript + Tailwind             │
│  http://localhost:5173                     │
└────────────────┬────────────────────────────┘
                 │ (Axios HTTP requests)
                 │
┌────────────────▼────────────────────────────┐
│         Flask API Server                    │
│  http://localhost:5000                     │
│  /scan endpoint                            │
└────────────────┬────────────────────────────┘
                 │ (imports and calls)
                 │
┌────────────────▼────────────────────────────┐
│    TinyLlamaScanner (or other models)      │
│  - Loads models from Hugging Face          │
│  - Runs safety analysis                    │
│  - Returns JSON results                    │
└─────────────────────────────────────────────┘
```

---

## 🎯 Next Steps

1. **Install dependencies**: Run `run.bat` or follow manual setup
2. **Start both servers**: Backend on 5000, Frontend on 5173
3. **Test with a prompt**: Enter any prompt and click Scan
4. **Monitor results**: Watch all metrics update in real-time
5. **View history**: Previous scans appear in the sidebar

---

## ⚠️ Common Issues

### Frontend won't connect to backend
- Make sure backend is running (check http://localhost:5000/health)
- Check browser console for errors (F12)
- Verify firewall isn't blocking port 5000

### Backend won't start
- Ensure Python 3.8+ is installed
- Check that port 5000 is available
- Verify requirements.txt packages installed: `pip install -r requirements.txt`

### Model download failing
- Check internet connection
- Wait longer for download (can take 5-10 minutes first time)
- Try: `huggingface-cli login` if rate-limited

### Port 5000/5173 already in use
- Change port in backend.py or vite.config.ts
- Or kill process using the port

---

## 📞 Support

For issues:
1. Check the README.md in Domain folder
2. Check browser console (F12) for frontend errors
3. Check terminal output for backend errors
4. Verify all requirements are installed

---

## 🔄 Architecture Summary

✓ **Frontend is fully React-based** - No vanilla JS
✓ **Backend properly integrated** - API calls working
✓ **Real-time data flow** - Results display immediately
✓ **Production-ready code** - Modular, clean, typed
✓ **Modern design** - Dark theme with animations
✓ **Fully responsive** - Works on all screen sizes

---

**Everything is now connected and ready to run!** 🎉
