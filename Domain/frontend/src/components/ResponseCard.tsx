interface ResponseCardProps {
  scan: {
    model: string;
    baseline_response: string;
    misbehaviour_score: number;
  };
}

function ResponseCard({ scan }: ResponseCardProps) {
  const getRiskLevel = (score: number) => {
    if (score >= 75) return { text: 'Critical', color: 'text-red-500', bg: 'bg-red-500/10' };
    if (score >= 50) return { text: 'High', color: 'text-orange-500', bg: 'bg-orange-500/10' };
    if (score >= 25) return { text: 'Moderate', color: 'text-yellow-500', bg: 'bg-yellow-500/10' };
    return { text: 'Low', color: 'text-green-500', bg: 'bg-green-500/10' };
  };

  const risk = getRiskLevel(scan.misbehaviour_score);

  return (
    <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
      <div className="flex justify-between items-start mb-4">
        <div>
          <p className="text-sm text-gray-400">Scan complete • {scan.model}</p>
          <p className="text-xs text-gray-500 mt-1">AI Model Safety Analysis</p>
        </div>
        <div className={`px-3 py-1 rounded-full text-sm font-medium ${risk.bg} ${risk.color}`}>
          {risk.text} Risk • {scan.misbehaviour_score.toFixed(1)}%
        </div>
      </div>
      <div className="mt-4 bg-gray-900 rounded-lg p-4 border border-gray-700">
        <p className="text-gray-300 text-sm leading-relaxed">{scan.baseline_response}</p>
      </div>
    </div>
  );
}

export default ResponseCard;