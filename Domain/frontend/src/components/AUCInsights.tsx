import type { MetricRow } from '../services/api';
import { formatMetric, getAUCColor, getAUCLabel } from './MetricsTable';

interface AUCInsightsProps {
  rows: MetricRow[];
}

function AUCInsights({ rows }: AUCInsightsProps) {
  const summaries = buildTaskSummaries(rows);

  return (
    <div className="space-y-6">
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <div className="flex flex-col gap-2 mb-4 sm:flex-row sm:items-start sm:justify-between">
          <h3 className="text-lg font-semibold">AUC Interpretation</h3>
          <span
            className="text-xs text-gray-400 border border-gray-700 rounded-full px-3 py-1"
            title="AUC represents the probability that the model ranks a misbehavior response higher than a normal one."
          >
            What does AUC mean?
          </span>
        </div>
        <p className="text-sm text-gray-300 leading-relaxed">
          AUC (Area Under Curve) measures how well the model separates normal and misbehavior responses.
          Higher AUC means better distinction between classes.
        </p>
        <div className="grid grid-cols-1 gap-3 mt-4 sm:grid-cols-4">
          {[
            ['>= 0.90', 'Excellent', 'text-green-500'],
            ['0.80-0.89', 'Good', 'text-green-400'],
            ['0.70-0.79', 'Moderate', 'text-yellow-500'],
            ['< 0.70', 'Poor', 'text-red-500'],
          ].map(([range, label, color]) => (
            <div key={label} className="bg-gray-900 rounded-lg p-3 border border-gray-700">
              <p className="text-xs text-gray-400">{range}</p>
              <p className={`text-sm font-semibold mt-1 ${color}`}>{label}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <h3 className="text-lg font-semibold mb-4">Performance Summary</h3>
        {summaries.length ? (
          <div className="space-y-3">
            {summaries.map((summary) => (
              <div key={summary.task} className="bg-gray-900 rounded-lg p-4 border border-gray-700">
                <p className="text-sm">
                  Best model for <span className="capitalize">{summary.task}</span>:{' '}
                  <span className="font-semibold text-green-500">{summary.best.model}</span>{' '}
                  <span className={getAUCColor(summary.best.auc)}>(AUC = {formatMetric(summary.best.auc)}, {getAUCLabel(summary.best.auc)})</span>
                </p>
                <p className="text-xs text-gray-400 mt-2">
                  Worst: <span className="text-red-500">{summary.worst.model}</span>{' '}
                  <span className={getAUCColor(summary.worst.auc)}>(AUC = {formatMetric(summary.worst.auc)})</span>
                </p>
              </div>
            ))}
          </div>
        ) : (
          <div className="bg-gray-900 rounded-lg p-4 border border-gray-700 text-sm text-gray-400">
            Metrics from the backend will appear here once AUC values are available.
          </div>
        )}
        <p className="text-sm text-gray-300 leading-relaxed mt-4">
          Models using internal feature-based detection (TinyLLaMA) may perform differently compared to pretrained classifier-based models.
        </p>
      </div>
    </div>
  );
}

function buildTaskSummaries(rows: MetricRow[]) {
  const withAuc = rows.filter((row): row is MetricRow & { auc: number } => row.auc !== undefined);
  const tasks = Array.from(new Set(withAuc.map((row) => row.task)));

  return tasks.map((task) => {
    const taskRows = withAuc.filter((row) => row.task === task).sort((a, b) => b.auc - a.auc);
    return {
      task,
      best: taskRows[0],
      worst: taskRows[taskRows.length - 1],
    };
  });
}

export default AUCInsights;
