interface MetricCardProps {
  title: string;
  value: number;
}

function MetricCard({ title, value }: MetricCardProps) {
  const getColor = (val: number) => {
    if (val >= 70) return 'text-red-500';
    if (val >= 40) return 'text-yellow-500';
    return 'text-green-500';
  };

  const getBgColor = (val: number) => {
    if (val >= 70) return 'bg-red-500/10';
    if (val >= 40) return 'bg-yellow-500/10';
    return 'bg-green-500/10';
  };

  return (
    <div className={`${getBgColor(value)} rounded-lg p-4 border border-gray-700`}>
      <p className="text-gray-400 text-sm">{title}</p>
      <p className={`text-3xl font-bold mt-2 ${getColor(value)}`}>{value.toFixed(1)}%</p>
    </div>
  );
}

export default MetricCard;