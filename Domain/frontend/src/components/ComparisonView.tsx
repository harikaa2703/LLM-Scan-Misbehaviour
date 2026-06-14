interface ComparisonViewProps {
  normalResponse: string;
  misbehaviorResponse: string;
}

function ComparisonView({ normalResponse, misbehaviorResponse }: ComparisonViewProps) {
  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
      <ResponsePanel title="Normal" response={normalResponse} />
      <ResponsePanel title="Misbehavior" response={misbehaviorResponse} />
    </div>
  );
}

function ResponsePanel({ title, response }: { title: string; response: string }) {
  return (
    <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
      <h3 className="text-lg font-semibold mb-4">{title}</h3>
      <div className="bg-gray-900 rounded-lg p-4 border border-gray-700 min-h-[160px]">
        <p className="text-gray-300 text-sm leading-relaxed whitespace-pre-wrap">{response}</p>
      </div>
    </div>
  );
}

export default ComparisonView;
