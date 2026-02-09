
import React, { useState, useCallback, useMemo, useEffect } from 'react';
import { GRID_POLYGONS, SVG_SIZE } from './constants';
import { 
  Undo2, 
  Redo2, 
  Copy, 
  Trash2, 
  Eye, 
  EyeOff, 
  Check, 
  Download,
  Terminal,
  Type
} from 'lucide-react';

const App: React.FC = () => {
  // Explicitly type the Set as Set<string> to prevent unknown inference
  const [activeIds, setActiveIds] = useState<Set<string>>(new Set<string>());
  const [history, setHistory] = useState<Set<string>[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [showGridLines, setShowGridLines] = useState(true);
  const [copied, setCopied] = useState(false);
  const [isExportVisible, setIsExportVisible] = useState(false);

  // Initialize history with an empty Set<string>
  useEffect(() => {
    if (history.length === 0) {
      setHistory([new Set<string>()]);
      setHistoryIndex(0);
    }
  }, []);

  const addToHistory = (newActiveIds: Set<string>) => {
    const newHistory = history.slice(0, historyIndex + 1);
    // Explicitly type the new history entry
    newHistory.push(new Set<string>(newActiveIds));
    setHistory(newHistory);
    setHistoryIndex(newHistory.length - 1);
  };

  const togglePolygon = useCallback((id: string) => {
    setActiveIds(prev => {
      // Fix inference issue by explicitly typing the new Set
      const next = new Set<string>(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      addToHistory(next);
      return next;
    });
  }, [history, historyIndex]);

  const undo = useCallback(() => {
    if (historyIndex > 0) {
      const prevIndex = historyIndex - 1;
      setHistoryIndex(prevIndex);
      // Ensure the set is correctly typed when restoring from history
      setActiveIds(new Set<string>(history[prevIndex]));
    }
  }, [history, historyIndex]);

  const redo = useCallback(() => {
    if (historyIndex < history.length - 1) {
      const nextIndex = historyIndex + 1;
      setHistoryIndex(nextIndex);
      // Ensure the set is correctly typed when restoring from history
      setActiveIds(new Set<string>(history[nextIndex]));
    }
  }, [history, historyIndex]);

  const reset = useCallback(() => {
    const next = new Set<string>();
    setActiveIds(next);
    addToHistory(next);
  }, [history, historyIndex]);

  const generatedSvgString = useMemo(() => {
    const activePolys = GRID_POLYGONS.filter(p => activeIds.has(p.id));
    const polygonsMarkup = activePolys.map(p => 
      `        <polygon id="${p.id}" points="${p.points}"/>`
    ).join('\n');

    return `<svg xmlns="http://www.w3.org/2000/svg" width="${SVG_SIZE}" height="${SVG_SIZE}" viewBox="0 0 ${SVG_SIZE} ${SVG_SIZE}">
    <!-- Background -->
    <rect x="0" y="0" width="${SVG_SIZE}" height="${SVG_SIZE}" fill="#fff"/>

    <!-- Active Shapes -->
    <g fill="#000" shape-rendering="crispEdges">
${polygonsMarkup}
    </g>
</svg>`;
  }, [activeIds]);

  const copyToClipboard = () => {
    navigator.clipboard.writeText(generatedSvgString).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  const downloadSvg = () => {
    const blob = new Blob([generatedSvgString], { type: 'image/svg+xml' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `typeface-grid-${Date.now()}.svg`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 text-slate-900 overflow-hidden">
      {/* Header */}
      <header className="h-16 border-b bg-white flex items-center justify-between px-6 shrink-0 shadow-sm z-10">
        <div className="flex items-center gap-2">
          <div className="p-2 bg-indigo-600 rounded-lg">
            <Type className="text-white w-5 h-5" />
          </div>
          <h1 className="text-xl font-bold tracking-tight bg-gradient-to-r from-indigo-600 to-indigo-400 bg-clip-text text-transparent">
            TypeGrid Designer
          </h1>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={undo}
            disabled={historyIndex <= 0}
            className="p-2 hover:bg-slate-100 disabled:opacity-30 disabled:hover:bg-transparent rounded-md transition-colors"
            title="Undo"
          >
            <Undo2 size={20} />
          </button>
          <button
            onClick={redo}
            disabled={historyIndex >= history.length - 1}
            className="p-2 hover:bg-slate-100 disabled:opacity-30 disabled:hover:bg-transparent rounded-md transition-colors"
            title="Redo"
          >
            <Redo2 size={20} />
          </button>
          <div className="w-px h-6 bg-slate-200 mx-1" />
          <button
            onClick={() => setShowGridLines(!showGridLines)}
            className={`p-2 rounded-md transition-colors ${showGridLines ? 'bg-indigo-50 text-indigo-600' : 'hover:bg-slate-100'}`}
            title="Toggle Grid Lines"
          >
            {showGridLines ? <Eye size={20} /> : <EyeOff size={20} />}
          </button>
          <button
            onClick={reset}
            className="p-2 hover:bg-red-50 text-slate-400 hover:text-red-500 rounded-md transition-colors"
            title="Reset All"
          >
            <Trash2 size={20} />
          </button>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 flex overflow-hidden">
        {/* Workspace */}
        <div className="flex-1 relative flex items-center justify-center bg-slate-200/30 overflow-auto p-12">
          <div className="bg-white p-12 shadow-2xl rounded-xl border border-slate-100 transform transition-transform duration-300">
            <svg
              width={SVG_SIZE * 2}
              height={SVG_SIZE * 2}
              viewBox={`0 0 ${SVG_SIZE} ${SVG_SIZE}`}
              className="cursor-pointer select-none"
              style={{ touchAction: 'none' }}
            >
              {/* Workspace Background */}
              <rect x="0" y="0" width={SVG_SIZE} height={SVG_SIZE} fill="#fff" />

              {/* All Polygons */}
              <g shape-rendering="crispEdges">
                {GRID_POLYGONS.map((poly) => (
                  <polygon
                    key={poly.id}
                    id={poly.id}
                    points={poly.points}
                    onClick={() => togglePolygon(poly.id)}
                    className={`transition-colors duration-150 cursor-pointer ${
                      activeIds.has(poly.id) 
                        ? 'fill-slate-900 stroke-slate-900' 
                        : 'fill-transparent stroke-transparent hover:fill-indigo-100/50'
                    }`}
                    strokeWidth="0.5"
                  />
                ))}
              </g>

              {/* Grid Lines */}
              {showGridLines && (
                <g fill="none" stroke="#1e90ff" strokeWidth="0.5" shape-rendering="crispEdges" opacity="0.6" pointerEvents="none">
                  {/* Vertical Lines */}
                  {Array.from({ length: 9 }).map((_, i) => (
                    <line key={`v${i}`} x1={i * 30} y1="0" x2={i * 30} y2={SVG_SIZE} />
                  ))}
                  {/* Horizontal Lines */}
                  {Array.from({ length: 9 }).map((_, i) => (
                    <line key={`h${i}`} x1="0" y1={i * 30} x2={SVG_SIZE} y2={i * 30} />
                  ))}
                  {/* Center Points / Diagonal Helper Visuals */}
                  {Array.from({ length: 8 }).map((_, r) => (
                    Array.from({ length: 8 }).map((_, c) => (
                      <circle key={`p${r}${c}`} cx={c * 30 + 15} cy={r * 30 + 15} r="0.5" fill="#1e90ff" />
                    ))
                  ))}
                </g>
              )}
            </svg>
          </div>
          
          {/* Legend/Context */}
          <div className="absolute bottom-6 left-6 text-xs text-slate-500 bg-white/80 backdrop-blur shadow px-3 py-2 rounded-lg border border-slate-200">
            <p>Active Polygons: <span className="font-bold text-slate-800">{activeIds.size}</span></p>
            <p>Click segments to toggle. Use history to undo mistakes.</p>
          </div>
        </div>

        {/* Side Panel (Code/Export) */}
        <aside className={`w-96 border-l bg-white flex flex-col transition-all duration-300 transform ${isExportVisible ? 'translate-x-0' : 'translate-x-full absolute right-0 inset-y-0 h-full'}`}>
          <div className="p-4 border-b flex items-center justify-between shrink-0">
            <div className="flex items-center gap-2">
              <Terminal size={18} className="text-indigo-600" />
              <h2 className="font-semibold text-sm uppercase tracking-wider text-slate-500">SVG Output</h2>
            </div>
            <button 
              onClick={() => setIsExportVisible(false)}
              className="p-1 hover:bg-slate-100 rounded text-slate-400"
            >
              Close
            </button>
          </div>
          
          <div className="flex-1 overflow-auto p-4 bg-slate-900 font-mono text-[10px] leading-relaxed text-indigo-300">
            <pre className="whitespace-pre-wrap select-all">
              {generatedSvgString}
            </pre>
          </div>

          <div className="p-4 bg-white border-t space-y-3 shrink-0">
            <button
              onClick={copyToClipboard}
              className={`w-full py-2.5 px-4 flex items-center justify-center gap-2 rounded-lg font-medium transition-all ${
                copied 
                ? 'bg-emerald-50 text-emerald-600 border border-emerald-200' 
                : 'bg-indigo-600 text-white hover:bg-indigo-700 active:scale-[0.98]'
              }`}
            >
              {copied ? <Check size={18} /> : <Copy size={18} />}
              {copied ? 'Copied!' : 'Copy SVG Markup'}
            </button>
            <button
              onClick={downloadSvg}
              className="w-full py-2.5 px-4 flex items-center justify-center gap-2 rounded-lg font-medium text-slate-700 bg-slate-100 hover:bg-slate-200 transition-all border border-slate-200"
            >
              <Download size={18} />
              Download .svg
            </button>
          </div>
        </aside>
      </main>

      {/* Persistent Call-to-Action / Floating Toggle for Panel */}
      {!isExportVisible && (
        <button
          onClick={() => setIsExportVisible(true)}
          className="fixed bottom-6 right-6 p-4 bg-indigo-600 text-white rounded-full shadow-2xl hover:bg-indigo-700 transition-all active:scale-90 group z-20"
        >
          <Terminal size={24} />
          <span className="absolute right-full mr-3 top-1/2 -translate-y-1/2 px-3 py-1 bg-slate-900 text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none">
            Export SVG
          </span>
        </button>
      )}
    </div>
  );
};

export default App;
