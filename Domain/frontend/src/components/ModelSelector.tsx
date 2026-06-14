interface ModelSelectorProps {
  selectedModel: string;
  onModelChange: (model: string) => void;
}

function ModelSelector({ selectedModel, onModelChange }: ModelSelectorProps) {
  return (
    <div className="flex items-center gap-4">
      <label className="text-sm font-medium">Model:</label>
      <select
        value={selectedModel}
        onChange={(e) => onModelChange(e.target.value)}
        className="p-2 bg-gray-800 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-llm-accent"
      >
        <option value="mistral">Ministral-3B</option>
        <option value="tinyllama">TinyLlama-1.1B</option>
        <option value="gpt2">GPT-2</option>
        <option value="gptneo">GPT-Neo</option>
      </select>
    </div>
  );
}

export default ModelSelector;
