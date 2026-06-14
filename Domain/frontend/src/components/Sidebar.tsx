import { motion } from 'framer-motion';

interface HistoryItem {
  id: string;
  prompt: string;
  model: string;
  risk: string;
  timestamp: Date;
}

interface SidebarProps {
  history: HistoryItem[];
  onDelete: (id: string) => void;
  onNewScan: () => void;
}

function Sidebar({ history, onDelete, onNewScan }: SidebarProps) {
  return (
    <div className="w-80 bg-gray-900 border-r border-gray-700 flex flex-col">
      <div className="p-4 border-b border-gray-700">
        <h1 className="text-xl font-bold text-llm-accent">LLMSCAN</h1>
        <button
          onClick={onNewScan}
          className="mt-4 w-full bg-llm-accent text-white py-2 rounded-lg hover:bg-blue-600 transition"
        >
          + New Scan
        </button>
      </div>
      <div className="flex-1 overflow-y-auto p-4">
        <h2 className="text-sm font-semibold mb-2">History</h2>
        {history.map(item => (
          <motion.div
            key={item.id}
            className="p-3 bg-gray-800 rounded-lg mb-2 cursor-pointer hover:bg-gray-700 transition relative group"
            whileHover={{ scale: 1.02 }}
          >
            <p className="text-sm truncate">{item.prompt}</p>
            <p className="text-xs text-gray-400">{item.model} • {item.risk}</p>
            <button
              onClick={(e) => {
                e.stopPropagation();
                onDelete(item.id);
              }}
              className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 text-red-400 hover:text-red-300 transition"
            >
              ✕
            </button>
          </motion.div>
        ))}
      </div>
    </div>
  );
}

export default Sidebar;
