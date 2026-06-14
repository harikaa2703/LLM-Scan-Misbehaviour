import type { CausalMapData, MisbehaviorTask } from '../services/api';

interface CausalMapProps {
  data?: CausalMapData;
  selectedTask: MisbehaviorTask;
  firstTokenOnly: boolean;
  loading?: boolean;
}

function CausalMap({ data, selectedTask, firstTokenOnly, loading = false }: CausalMapProps) {
  const hasRawHeatmap = Boolean(data?.tokens?.length && data.layers?.length && data.values?.length);

  return (
    <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
      <div className="flex flex-col gap-1 mb-4 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <h3 className="text-lg font-semibold">Causal Map</h3>
          <p className="text-sm text-gray-400 capitalize">
            {selectedTask} - {firstTokenOnly ? 'First token only' : 'Full response'}
          </p>
        </div>
        {loading && <span className="text-sm text-gray-400">Updating...</span>}
      </div>

      {data?.image ? (
        <img src={data.image} alt={`${selectedTask} causal map`} className="w-full h-auto rounded border border-gray-700" />
      ) : hasRawHeatmap && data ? (
        <RawHeatmap data={data} />
      ) : (
        <div className="bg-gray-900 rounded-lg p-6 border border-gray-700 text-center text-gray-400">
          Run a scan to view the causal map.
        </div>
      )}
    </div>
  );
}

function RawHeatmap({ data }: { data: CausalMapData }) {
  const tokens = data.tokens ?? [];
  const layers = data.layers ?? [];
  const values = data.values ?? [];
  const columns = `96px repeat(${Math.max(tokens.length, 1)}, minmax(32px, 1fr))`;

  return (
    <div className="overflow-x-auto">
      <div className="min-w-[640px]">
        <div className="grid gap-1 text-xs text-gray-400" style={{ gridTemplateColumns: columns }}>
          <div />
          {tokens.map((token, index) => (
            <div key={`${token}-${index}`} className="truncate text-center" title={token}>
              {token}
            </div>
          ))}
          {layers.map((layer, layerIndex) => (
            <HeatmapRow
              key={layer}
              layer={layer}
              values={values[layerIndex] ?? []}
              columns={tokens.length}
            />
          ))}
        </div>
        <div className="flex items-center justify-end gap-2 mt-4 text-xs text-gray-400">
          <span>Low</span>
          <div className="h-2 w-32 rounded-full bg-gradient-to-r from-green-500 via-yellow-500 to-red-500" />
          <span>High</span>
        </div>
      </div>
    </div>
  );
}

function HeatmapRow({ layer, values, columns }: { layer: string; values: number[]; columns: number }) {
  return (
    <>
      <div className="flex items-center pr-2 text-right text-gray-400">{layer}</div>
      {Array.from({ length: columns }).map((_, index) => {
        const value = values[index] ?? 0;
        return (
          <div
            key={`${layer}-${index}`}
            className="h-8 rounded border border-gray-900"
            title={`${layer}: ${value.toFixed(3)}`}
            style={{ backgroundColor: heatColor(value) }}
          />
        );
      })}
    </>
  );
}

function heatColor(value: number) {
  const clamped = Math.max(0, Math.min(1, value));
  if (clamped < 0.5) {
    const ratio = clamped / 0.5;
    return `rgb(${Math.round(34 + ratio * 200)}, ${Math.round(197 + ratio * 23)}, ${Math.round(94 - ratio * 54)})`;
  }

  const ratio = (clamped - 0.5) / 0.5;
  return `rgb(${Math.round(234 + ratio * 5)}, ${Math.round(179 - ratio * 111)}, ${Math.round(8 - ratio * 8)})`;
}

export default CausalMap;
