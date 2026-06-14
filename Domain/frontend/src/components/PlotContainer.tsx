import { motion } from 'framer-motion';

interface PlotContainerProps {
  plots: {
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

function PlotContainer({ plots }: PlotContainerProps) {
  const plotItems = [
    { key: 'layer_influence', title: 'Layer Influence' },
    { key: 'misbehaviour_scores', title: 'Misbehaviour Scores' },
    { key: 'token_confidence', title: 'Token Confidence' },
    { key: 'activation_heatmap', title: 'Activation Heatmap' },
    { key: 'safe_vs_unsafe', title: 'Safe vs Unsafe Activation' },
    { key: 'consistency', title: 'Consistency Check' },
    { key: 'hallucination_similarity', title: 'Hallucination Similarity' },
    { key: 'before_after_intervention', title: 'Before vs After Intervention' },
  ];

  return (
    <div className="space-y-6">
      {plotItems.map((item, index) => {
        const plotData = plots[item.key as keyof typeof plots];
        if (!plotData && item.key !== 'safe_vs_unsafe') return null; // safe_vs_unsafe is optional
        if (item.key === 'safe_vs_unsafe' && !plotData) return null;

        return (
          <motion.div
            key={item.key}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className="bg-gray-800 rounded-lg p-6 border border-gray-700"
          >
            <h3 className="text-lg font-semibold mb-4">{item.title}</h3>
            <img
              src={plotData}
              alt={item.title}
              className="w-full h-auto rounded"
            />
          </motion.div>
        );
      })}
    </div>
  );
}

export default PlotContainer;
