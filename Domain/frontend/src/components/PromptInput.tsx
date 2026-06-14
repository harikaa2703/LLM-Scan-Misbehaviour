import { useState } from 'react';
import { motion } from 'framer-motion';

interface PromptInputProps {
  onSubmit: (prompt: string) => void;
  loading: boolean;
}

function PromptInput({ onSubmit, loading }: PromptInputProps) {
  const [input, setInput] = useState('');

  const handleSubmit = () => {
    if (input.trim()) {
      onSubmit(input);
      setInput('');
    }
  };

  return (
    <div className="p-4 border-t border-gray-700 bg-gray-900">
      <div className="flex gap-3">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && !loading && handleSubmit()}
          placeholder="Enter a prompt to scan..."
          disabled={loading}
          className="flex-1 bg-gray-800 border border-gray-600 rounded-lg p-3 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-llm-accent"
        />
        <motion.button
          onClick={handleSubmit}
          disabled={loading}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          className="px-6 bg-llm-accent text-white rounded-lg hover:bg-blue-600 disabled:bg-gray-600 transition"
        >
          {loading ? 'Scanning...' : 'Scan'}
        </motion.button>
      </div>
    </div>
  );
}

export default PromptInput;