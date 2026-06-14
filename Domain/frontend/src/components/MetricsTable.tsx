import type { MetricRow } from '../services/api';

interface MetricsTableProps {
  rows: MetricRow[];
}

export function getAUCLabel(auc?: number) {
  if (auc === undefined) return 'Unavailable';
  if (auc >= 0.9) return 'Excellent';
  if (auc >= 0.8) return 'Good';
  if (auc >= 0.7) return 'Moderate';
  return 'Poor';
}

export function getAUCColor(auc?: number) {
  if (auc === undefined) return 'text-gray-400';
  if (auc > 0.85) return 'text-green-500';
  if (auc >= 0.7) return 'text-yellow-500';
  return 'text-red-500';
}

export function formatMetric(value?: number) {
  if (value === undefined) return 'Not stored';
  return value <= 1 ? value.toFixed(2) : `${value.toFixed(1)}%`;
}

function MetricsTable({ rows }: MetricsTableProps) {
  const taskRows = rows.filter((row) => row.auc !== undefined);

  return (
    <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
      <h3 className="text-lg font-semibold mb-4">Evaluation Metrics</h3>
      {taskRows.length ? (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-700 text-left text-gray-400">
                <th className="py-3 pr-4 font-medium">Model</th>
                <th className="py-3 pr-4 font-medium">Task</th>
                <th className="py-3 pr-4 font-medium">AUC</th>
                <th className="py-3 pr-4 font-medium">Accuracy</th>
              </tr>
            </thead>
            <tbody>
              {taskRows.map((row) => (
                <tr key={`${row.model}-${row.task}`} className="border-b border-gray-700/60 last:border-0">
                  <td className="py-3 pr-4">{row.model}</td>
                  <td className="py-3 pr-4 capitalize">{row.task}</td>
                  <td className={`py-3 pr-4 font-medium ${getAUCColor(row.auc)}`}>
                    {formatMetric(row.auc)} <span className="text-xs text-gray-400">({getAUCLabel(row.auc)})</span>
                    <div className="h-1.5 bg-gray-900 rounded-full overflow-hidden mt-2 max-w-[160px]">
                      <div
                        className={`h-full ${getAUCBarColor(row.auc)}`}
                        style={{ width: `${Math.max(0, Math.min(100, (row.auc ?? 0) * 100))}%` }}
                      />
                    </div>
                  </td>
                  <td className="py-3 pr-4">{formatMetric(row.accuracy)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="bg-gray-900 rounded-lg p-4 border border-gray-700 text-sm text-gray-400">
          No AUC values are available from the backend yet.
        </div>
      )}
    </div>
  );
}

function getAUCBarColor(auc?: number) {
  if (auc === undefined) return 'bg-gray-700';
  if (auc > 0.85) return 'bg-green-500';
  if (auc >= 0.7) return 'bg-yellow-500';
  return 'bg-red-500';
}

export default MetricsTable;
