import React from 'react';
import { X, BarChart3, AlertCircle } from 'lucide-react';
import { ModelBenchmarkItem } from '../types';

interface BenchmarkModalProps {
  isOpen: boolean;
  onClose: () => void;
  benchmarks: ModelBenchmarkItem[];
}

export const BenchmarkModal: React.FC<BenchmarkModalProps> = ({
  isOpen,
  onClose,
  benchmarks,
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-fade-in">
      <div className="cyber-card w-full max-w-4xl max-h-[90vh] overflow-y-auto p-6 border-cyan-500/40 shadow-[0_0_30px_rgba(0,240,255,0.15)] relative">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-[#1f293d] pb-4 mb-4">
          <div className="flex items-center gap-2.5">
            <BarChart3 className="h-5 w-5 text-cyan-400" />
            <h2 className="text-sm font-bold font-mono tracking-wider text-slate-100 uppercase">
              Master Model Benchmark Comparison (Phases 2, 3, and 4)
            </h2>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-slate-200 transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Scientific Context Banner */}
        <div className="p-3.5 rounded-xl bg-[#0a0d14] border border-cyan-500/30 text-xs text-slate-300 mb-5 flex items-start gap-3">
          <AlertCircle className="h-4 w-4 text-cyan-400 flex-shrink-0 mt-0.5" />
          <div>
            <strong className="text-cyan-300">Scientific Context & Architectural Rationale:</strong>
            <p className="mt-1 text-slate-400 leading-relaxed">
              Random Forest provides the highest discriminative metric (<code className="text-cyan-300">T+1 ROC-AUC = 0.6631</code>) on the held-out test partition due to ensemble decision boundaries on aggregated statistical moments. The Latent Network State World Model delivers the core system innovation: <strong>5-step recursive latent-state forecasting</strong> (S_t → S_t+1 ... S_t+5) without teacher forcing or multi-head divergence.
            </p>
          </div>
        </div>

        {/* Benchmark Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead className="text-[10px] uppercase text-slate-400 border-b border-[#1f293d] bg-[#0a0d14]">
              <tr>
                <th className="py-2.5 px-3">Model Architecture</th>
                <th className="py-2.5 px-3">Type</th>
                <th className="py-2.5 px-3">T+1 ROC-AUC</th>
                <th className="py-2.5 px-3">T+1 PR-AUC</th>
                <th className="py-2.5 px-3">T+1 Macro F1</th>
                <th className="py-2.5 px-3">T+5 ROC-AUC</th>
                <th className="py-2.5 px-3">Brier Score</th>
                <th className="py-2.5 px-3">False Pos Rate</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#182030] text-slate-200">
              {benchmarks.map((m, idx) => {
                const isWM = m.name.includes('World Model');
                const isRF = m.name.includes('Random Forest');

                return (
                  <tr
                    key={idx}
                    className={`hover:bg-[#161c2d]/80 transition-colors ${
                      isWM ? 'bg-cyan-950/20 font-bold text-cyan-200' : isRF ? 'bg-emerald-950/20 font-semibold' : ''
                    }`}
                  >
                    <td className="py-2.5 px-3 whitespace-nowrap">
                      {m.name}
                      {isWM && <span className="ml-2 text-[9px] px-1.5 py-0.5 rounded bg-cyan-500/20 text-cyan-300 border border-cyan-500/40">CORE INNOVATION</span>}
                      {isRF && <span className="ml-2 text-[9px] px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/40">TOP BENCHMARK</span>}
                    </td>
                    <td className="py-2.5 px-3 text-slate-400">{m.type}</td>
                    <td className="py-2.5 px-3">{m.t1_roc_auc.toFixed(4)}</td>
                    <td className="py-2.5 px-3">{m.t1_pr_auc.toFixed(4)}</td>
                    <td className="py-2.5 px-3">{m.t1_macro_f1.toFixed(4)}</td>
                    <td className="py-2.5 px-3">{m.t5_roc_auc.toFixed(4)}</td>
                    <td className="py-2.5 px-3">{m.brier_score.toFixed(4)}</td>
                    <td className="py-2.5 px-3">{(m.fpr * 100).toFixed(2)}%</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <div className="mt-5 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-lg bg-[#161c2d] hover:bg-[#1f293d] border border-[#1f293d] text-xs font-mono text-slate-200 transition-colors"
          >
            Close Benchmarks
          </button>
        </div>
      </div>
    </div>
  );
};
