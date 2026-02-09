
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
  Type,
  RefreshCw
} from 'lucide-react';

const App: React.FC = () => {
  // Helper to get all IDs
  const allIdsSet = useMemo(() => new Set<string>(GRID_POLYGONS.map(p => p.id)), []);

  // Initialize state with all polygons active
  const [activeIds, setActiveIds] = useState<Set<string>>(allIdsSet);
  const [history, setHistory] = useState<Set<string>[]>([new Set<string>(allIdsSet)]);
  const [historyIndex, setHistoryIndex] = useState(0);
  
  const [showGridLines, setShowGridLines] = useState(true);
  const [copied, setCopied] = useState(false);
  const [isExportVisible, setIsExportVisible] = useState(true);

  const addToHistory = (newActiveIds: Set<string>) => {
    const newHistory = history.slice(0, historyIndex + 1);
    newHistory.push(new Set<string>(newActiveIds));
    setHistory(newHistory);
    setHistoryIndex(newHistory.length - 1);
  };

  const togglePolygon = useCallback((id: string) => {
    setActiveIds(prev => {
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
      setActiveIds(new Set<string>(history[prevIndex]));
    }
  }, [history, historyIndex]);

  const redo = useCallback(() => {
    if (historyIndex < history.length - 1) {
      const nextIndex = historyIndex + 1;
      setHistoryIndex(nextIndex);
      setActiveIds(new Set<string>(history[nextIndex]));
    }
  }, [history, historyIndex]);

  const reset = useCallback(() => {
    const next = new Set<string>();
    setActiveIds(next);
    addToHistory(next);
  }, [history, historyIndex]);

  const fillAll = useCallback(() => {
    setActiveIds(allIdsSet);
    addToHistory(allIdsSet);
  }, [allIdsSet, history, historyIndex]);

  const invertAll = useCallback(() => {
    setActiveIds(prev => {
      const next = new Set<string>();
      GRID_POLYGONS.forEach(poly => {
        if (!prev.has(poly.id)) {
          next.add(poly.id);
        }
      });
      addToHistory(next);
      return next;
    });
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

        <div className="flex items-center gap-2">
          <button
            onClick={undo}
            disabled={historyIndex <= 0}
            className="p-2 hover:bg-slate-100 disabled:opacity-30 disabled:hover:bg-transparent rounded-md transition-colors"
            title="Undo"
          >
            <Undo2 size={18} />
          </button>
          <button
            onClick={redo}
            disabled={historyIndex >= history.length - 1}
            className="p-2 hover:bg-slate-100 disabled:opacity-30 disabled:hover:bg-transparent rounded-md transition-colors"
            title="Redo"
          >
            <Redo2 size={18} />
          </button>
          <div className="w-px h-5 bg-slate-200 mx-1" />
          <button
            onClick={() => setShowGridLines(!showGridLines)}
            className={`p-2 rounded-md transition-colors ${showGridLines ? 'bg-indigo-50 text-indigo-600' : 'hover:bg-slate-100'}`}
            title="Toggle Grid Lines"
          >
            {showGridLines ? <Eye size={18} /> : <EyeOff size={18} />}
          </button>
          <div className="w-px h-5 bg-slate-200 mx-1" />
          <button
            onClick={fillAll}
            className="p-2 hover:bg-indigo-50 text-slate-500 hover:text-indigo-600 rounded-md transition-colors"
            title="Select All"
          >
            <Check size={18} />
          </button>
          <button
            onClick={invertAll}
            className="p-2 hover:bg-indigo-50 text-slate-500 hover:text-indigo-600 rounded-md transition-colors"
            title="Invert Selection"
          >
            <RefreshCw size={18} />
          </button>
          <button
            onClick={reset}
            className="p-2 hover:bg-red-50 text-slate-500 hover:text-red-500 rounded-md transition-colors"
            title="Clear All"
          >
            <Trash2 size={18} />
          </button>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 flex overflow-hidden">
        {/* Workspace - Aligned to top-center */}
        <div className="flex-1 relative flex items-start justify-center bg-slate-100/50 overflow-auto pt-16 pb-24 transition-all">
          <div className="bg-white p-12 shadow-xl rounded-xl border border-slate-200">
            <svg
              width={SVG_SIZE * 2}
              height={SVG_SIZE * 2}
              viewBox={`0 0 ${SVG_SIZE} ${SVG_SIZE}`}
              className="cursor-pointer select-none"
              style={{ touchAction: 'none' }}
            >
              <rect x="0" y="0" width={SVG_SIZE} height={SVG_SIZE} fill="#fff" />

              <g shape-rendering="crispEdges">
                {GRID_POLYGONS.map((poly) => (
                  <polygon
                    key={poly.id}
                    id={poly.id}
                    points={poly.points}
                    onClick={() => togglePolygon(poly.id)}
                    className={`transition-colors duration-100 cursor-pointer ${
                      activeIds.has(poly.id) 
                        ? 'fill-slate-900 stroke-slate-900' 
                        : 'fill-transparent stroke-transparent hover:fill-indigo-100/50'
                    }`}
                    strokeWidth="0.1"
                  />
                ))}
              </g>

              {showGridLines && (
                <g fill="none" stroke="#1e90ff" strokeWidth="0.5" shape-rendering="crispEdges" opacity="0.4" pointerEvents="none">
                  {Array.from({ length: 9 }).map((_, i) => (
                    <line key={`v${i}`} x1={i * 30} y1="0" x2={i * 30} y2={SVG_SIZE} />
                  ))}
                  {Array.from({ length: 9 }).map((_, i) => (
                    <line key={`h${i}`} x1="0" y1={i * 30} x2={SVG_SIZE} y2={i * 30} />
                  ))}
                </g>
              )}
            </svg>
          </div>
          
          <div className="absolute bottom-6 left-6 text-xs text-slate-400 font-medium tracking-wider">
            GRID: 8x8 TRIANGULAR SUBDIVISIONS | ACTIVE: {activeIds.size} / 256
          </div>
        </div>

        {/* Side Panel (Code/Export) - Persistent */}
        <aside className={`flex flex-col border-l bg-white transition-all duration-300 ${isExportVisible ? 'w-[450px]' : 'w-0 overflow-hidden'}`}>
          <div className="p-4 border-b flex items-center justify-between shrink-0 bg-slate-50/50">
            <div className="flex items-center gap-2">
              <Terminal size={18} className="text-indigo-600" />
              <h2 className="font-bold text-xs uppercase tracking-widest text-slate-500">Live SVG Markup</h2>
            </div>
            <button 
              onClick={() => setIsExportVisible(false)}
              className="p-1.5 hover:bg-slate-200 rounded text-slate-400 transition-colors"
              title="Collapse Panel"
            >
              <Terminal size={16} />
            </button>
          </div>
          
          <div className="flex-1 overflow-auto p-6 bg-slate-950 font-mono text-[11px] leading-relaxed text-indigo-200/90 selection:bg-indigo-500/30">
            <pre className="whitespace-pre-wrap">
              {generatedSvgString}
            </pre>
          </div>

          <div className="p-6 bg-white border-t border-slate-100 space-y-3 shrink-0 shadow-[0_-4px_6px_-1px_rgba(0,0,0,0.05)]">
            <button
              onClick={copyToClipboard}
              className={`w-full py-3 px-4 flex items-center justify-center gap-2 rounded-xl font-semibold transition-all shadow-sm ${
                copied 
                ? 'bg-emerald-50 text-emerald-600 border border-emerald-200' 
                : 'bg-indigo-600 text-white hover:bg-indigo-700 active:scale-[0.98] hover:shadow-indigo-200 hover:shadow-lg'
              }`}
            >
              {copied ? <Check size={18} /> : <Copy size={18} />}
              {copied ? 'Copied to Clipboard' : 'Copy SVG Markup'}
            </button>
            <button
              onClick={downloadSvg}
              className="w-full py-3 px-4 flex items-center justify-center gap-2 rounded-xl font-semibold text-slate-600 bg-white hover:bg-slate-50 transition-all border border-slate-200 active:scale-[0.98]"
            >
              <Download size={18} />
              Download SVG File
            </button>
          </div>
        </aside>
      </main>

      {/* Floating Toggle if Panel is Hidden */}
      {!isExportVisible && (
        <button
          onClick={() => setIsExportVisible(true)}
          className="fixed bottom-8 right-8 p-5 bg-indigo-600 text-white rounded-full shadow-2xl hover:bg-indigo-700 transition-all active:scale-90 z-20"
        >
          <Terminal size={24} />
        </button>
      )}
    </div>
  );
};

export default App;
