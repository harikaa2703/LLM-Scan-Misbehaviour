interface ScoreBarProps {
  scores: {
    hallucination: number;
    misbehaviour: number;
    toxicity: number;
    bias: number;
    jailbreak: number;
  };
}

function ScoreBar({ scores }: ScoreBarProps) {
  const getBarColor = (value: number) => {
    if (value >= 70) return 'bg-red-500';
    if (value >= 40) return 'bg-yellow-500';
    return 'bg-green-500';
  };

  const metrics = [
    { label: 'Hallucination', value: scores.hallucination },
    { label: 'Misbehaviour', value: scores.misbehaviour },
    { label: 'Toxicity', value: scores.toxicity },
    { label: 'Bias', value: scores.bias },
    { label: 'Jailbreak', value: scores.jailbreak },
  ];

  return (
    <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
      <h3 className="text-lg font-semibold mb-4">Score Breakdown</h3>
      {metrics.map((metric) => (
        <div key={metric.label} className="mb-4">
          <div className="flex justify-between mb-2">
            <span className="text-sm text-gray-400">{metric.label}</span>
            <span className="text-sm font-medium">{metric.value.toFixed(1)}%</span>
          </div>
          <div className="h-2 bg-gray-900 rounded-full overflow-hidden">
            <div
              className={`h-full ${getBarColor(metric.value)} transition-all duration-300`}
              style={{ width: `${Math.max(0, Math.min(100, metric.value))}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

export default ScoreBar;
