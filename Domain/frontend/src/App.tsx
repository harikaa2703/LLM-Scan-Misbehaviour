import { useState } from 'react';
import { motion } from 'framer-motion';
import Sidebar from './components/Sidebar';
import PromptInput from './components/PromptInput';
import ModelSelector from './components/ModelSelector';
import ResponseCard from './components/ResponseCard';
import MetricCard from './components/MetricCard';
import ScoreBar from './components/ScoreBar';
import LayerChart from './components/LayerChart';
import PlotContainer from './components/PlotContainer';
import CausalMap from './components/CausalMap';
import ComparisonView from './components/ComparisonView';
import MetricsTable from './components/MetricsTable';
import AUCInsights from './components/AUCInsights';
import {
  fetchEvaluationMetrics,
  fetchCausalMap,
  normalizeCausalMap,
  normalizeMetricRows,
  runScan,
  type CausalMapData,
  type MetricRow,
  type MisbehaviorTask,
} from './services/api';

interface ScanResult {
  model: string;
  prompt: string;
  baseline_response: string;
  normal_response: string;
  misbehavior_response: string;
  hallucination_score: number;
  selfcheck_score: number;
  rf_score: number;
  toxicity_score: number;
  jailbreak_score: number;
  bias_score: number;
  entropy: number;
  misbehaviour_score: number;
  layer_influence: number[];
  causal_map?: CausalMapData;
  metrics: MetricRow[];
  plots?: {
    layer_influence?: string;
    misbehaviour_scores?: string;
    token_confidence?: string;
    activation_heatmap?: string;
    safe_vs_unsafe?: string;
    consistency?: string;
    hallucination_similarity?: string;
    before_after_intervention?: string;
  };
}

interface HistoryItem {
  id: string;
  prompt: string;
  model: string;
  risk: string;
  timestamp: Date;
}

function App() {
  const [selectedModel, setSelectedModel] = useState('tinyllama');
  const [selectedTask, setSelectedTask] = useState<MisbehaviorTask>('toxicity');
  const [firstTokenOnly, setFirstTokenOnly] = useState(false);
  const [currentScan, setCurrentScan] = useState<ScanResult | null>(null);
  const [causalMap, setCausalMap] = useState<CausalMapData | undefined>();
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [causalMapLoading, setCausalMapLoading] = useState(false);

  const handleDeleteHistory = (id: string) => {
    setHistory(prev => prev.filter(item => item.id !== id));
  };

  const handleNewScan = () => {
    setCurrentScan(null);
    setCausalMap(undefined);
    setLoading(false);
  };

  const updateCausalMap = async (
    prompt: string,
    model: string,
    task: MisbehaviorTask,
    firstTokenMode: boolean,
    fallback?: CausalMapData,
  ) => {
    setCausalMapLoading(true);
    const updatedMap = await fetchCausalMap(prompt, model, task, firstTokenMode);
    setCausalMap(updatedMap ?? fallback);
    setCausalMapLoading(false);
  };

  const handleTaskChange = (task: MisbehaviorTask) => {
    setSelectedTask(task);
    if (currentScan) {
      void updateCausalMap(currentScan.prompt, currentScan.model, task, firstTokenOnly, currentScan.causal_map);
    }
  };

  const handleFirstTokenToggle = () => {
    const nextValue = !firstTokenOnly;
    setFirstTokenOnly(nextValue);
    if (currentScan) {
      void updateCausalMap(currentScan.prompt, currentScan.model, selectedTask, nextValue, currentScan.causal_map);
    }
  };

  const handleScan = async (prompt: string) => {
    setCurrentScan(null); // Clear previous scan results
    setCausalMap(undefined);
    setLoading(true);
    try {
      const scanResult = await runScan(prompt, selectedModel);
      const fallbackCausalMap = normalizeCausalMap(scanResult.causal_map)
        ?? normalizeCausalMap(scanResult.plots?.activation_heatmap);
      let metricRows = normalizeMetricRows(scanResult.evaluation_metrics ?? scanResult.metrics);
      if (!metricRows.length) {
        metricRows = await fetchEvaluationMetrics();
      }
      
      // Extract data from the scan result
      const baselineScores = scanResult.baseline_scores ?? {};
      const scoreSummary = scanResult.score_summary ?? {};
      const bestIntervention = getBestInterventionResponse(scanResult.interventions);
      const normalResponse = scanResult.normal_response || scanResult.baseline_response || 'No response available';
      const misbehaviorResponse = scanResult.misbehavior_response
        || scanResult.unsafe_response
        || bestIntervention
        || 'No intervention response available';
      const result: ScanResult = {
        model: selectedModel,
        prompt,
        baseline_response: normalResponse,
        normal_response: normalResponse,
        misbehavior_response: misbehaviorResponse,
        hallucination_score: scoreFrom(scanResult, 'hallucination_score', 'hallucination'),
        selfcheck_score: scanResult.hallucination_details?.low_agreement_count || 0,
        rf_score: 0, // not available in current response
        toxicity_score: toPercentScore(scoreSummary.toxicity ?? baselineScores.toxicity_score),
        jailbreak_score: toPercentScore(scoreSummary.jailbreak ?? baselineScores.jailbreak_score),
        bias_score: toPercentScore(scoreSummary.bias ?? baselineScores.bias_score),
        entropy: toNumber(baselineScores.entropy),
        misbehaviour_score: toPercentScore(scoreSummary.misbehaviour ?? baselineScores.misbehaviour_score),
        layer_influence: scanResult.layer_analysis?.influence_scores || [],
        plots: scanResult.plots,
        causal_map: fallbackCausalMap,
        metrics: metricRows,
      };
      
      setCurrentScan(result);
      setCausalMap(fallbackCausalMap);
      void updateCausalMap(prompt, selectedModel, selectedTask, firstTokenOnly, fallbackCausalMap);
      const riskLevel = result.misbehaviour_score > 50 ? 'High' : result.misbehaviour_score > 25 ? 'Moderate' : 'Low';
      setHistory(prev => [...prev, {
        id: Date.now().toString(),
        prompt,
        model: selectedModel,
        risk: riskLevel,
        timestamp: new Date(),
      }]);
    } catch (error) {
      console.error('Scan error:', error);
      alert('Error running scan. Make sure backend is running on http://localhost:5000');
    }
    setLoading(false);
  };

  return (
    <div className="flex h-screen bg-llm-dark text-white">
      <Sidebar history={history} onDelete={handleDeleteHistory} onNewScan={handleNewScan} />
      <div className="flex-1 flex flex-col">
        <div className="p-4 border-b border-gray-700">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <ModelSelector selectedModel={selectedModel} onModelChange={setSelectedModel} />
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
              <label className="flex items-center gap-3 text-sm">
                <span className="text-gray-400">Misbehavior:</span>
                <select
                  value={selectedTask}
                  onChange={(event) => handleTaskChange(event.target.value as MisbehaviorTask)}
                  className="p-2 bg-gray-800 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-llm-accent capitalize"
                >
                  <option value="toxicity">toxicity</option>
                  <option value="jailbreak">jailbreak</option>
                  <option value="lie">lie</option>
                  <option value="backdoor">backdoor</option>
                </select>
              </label>
              <button
                type="button"
                onClick={handleFirstTokenToggle}
                className="flex items-center gap-3 text-sm text-gray-300"
              >
                <span>First token only mode</span>
                <span className={`relative h-6 w-11 rounded-full border transition ${firstTokenOnly ? 'bg-llm-accent border-llm-accent' : 'bg-gray-800 border-gray-600'}`}>
                  <span className={`absolute top-0.5 h-5 w-5 rounded-full bg-white transition ${firstTokenOnly ? 'left-5' : 'left-0.5'}`} />
                </span>
              </button>
            </div>
          </div>
        </div>
        <div className="flex-1 overflow-y-auto p-4">
          {loading ? (
            <div className="text-center text-gray-300">Running scan... please wait</div>
          ) : currentScan ? (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="space-y-6"
            >
              <ResponseCard scan={currentScan} />
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-5">
                <MetricCard title="Hallucination" value={currentScan.hallucination_score} />
                <MetricCard title="Toxicity" value={currentScan.toxicity_score} />
                <MetricCard title="Bias" value={currentScan.bias_score} />
                <MetricCard title="Jailbreak" value={currentScan.jailbreak_score} />
                <MetricCard title="Entropy" value={currentScan.entropy} />
              </div>
              <ScoreBar scores={{
                hallucination: currentScan.hallucination_score,
                misbehaviour: currentScan.misbehaviour_score,
                toxicity: currentScan.toxicity_score,
                bias: currentScan.bias_score,
                jailbreak: currentScan.jailbreak_score,
              }} />
              <ComparisonView
                normalResponse={currentScan.normal_response}
                misbehaviorResponse={currentScan.misbehavior_response}
              />
              <CausalMap
                data={causalMap}
                selectedTask={selectedTask}
                firstTokenOnly={firstTokenOnly}
                loading={causalMapLoading}
              />
              <LayerChart data={currentScan.layer_influence} />
              <MetricsTable rows={currentScan.metrics} />
              <AUCInsights rows={currentScan.metrics} />
              {currentScan.plots && <PlotContainer plots={currentScan.plots} />}
            </motion.div>
          ) : (
            <div className="text-center text-gray-500">No scan yet. Enter a prompt and click Run.</div>
          )}
        </div>
        <PromptInput onSubmit={handleScan} loading={loading} />
      </div>
    </div>
  );
}

function toNumber(value: unknown) {
  const numberValue = Number(typeof value === 'string' ? value.replace('%', '') : value);
  return Number.isFinite(numberValue) ? numberValue : 0;
}

function toPercentScore(value: unknown) {
  const numberValue = toNumber(value);
  if (numberValue > 0 && numberValue <= 1) {
    return numberValue * 100;
  }
  return numberValue;
}

function scoreFrom(scanResult: { baseline_scores?: Record<string, number | string>; score_summary?: Record<string, number | string> }, scoreKey: string, summaryKey: string) {
  return toPercentScore(scanResult.score_summary?.[summaryKey] ?? scanResult.baseline_scores?.[scoreKey]);
}

function getBestInterventionResponse(interventions?: Array<{ new_response?: string; scores?: Record<string, number | string> }>) {
  if (!interventions?.length) return undefined;

  return interventions
    .filter((intervention) => intervention.new_response)
    .sort((a, b) => toNumber(b.scores?.semantic_drift_score) - toNumber(a.scores?.semantic_drift_score))[0]
    ?.new_response;
}

export default App;
