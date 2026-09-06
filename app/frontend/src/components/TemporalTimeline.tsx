import React from 'react';
import { History } from 'lucide-react';

interface TemporalTimelineProps {
  importance: number[];
  mostInfluentialStep: string;
}

export const TemporalTimeline: React.FC<TemporalTimelineProps> = ({
  importance,
  mostInfluentialStep,
}) => {
  // Ensure array has 20 elements
  const steps = importance && importance.length === 20 ? importance : Array(20).fill(0.05);

  return (
    <div className="cyber-card p-5">
      <div className="flex items-center justify-between border-b border-[#1f293d] pb-3 mb-4">
        <div className="flex items-center gap-2">
          <History className="h-4 w-4 text-cyan-400" />
          <h2 className="text-xs font-mono uppercase tracking-wider text-slate-200 font-bold">
            20-Flow Temporal Memory Saliency (W=20)
          </h2>
        </div>
        <div className="text-[10px] font-mono text-cyan-300 bg-cyan-950/40 px-2 py-0.5 rounded border border-cyan-800/40">
          Peak Trigger: {mostInfluentialStep || 't-0'}
        </div>
      </div>

      {/* Bar timeline of 20 historical flows */}
      <div className="flex items-end gap-1 sm:gap-1.5 h-24 pt-2">
        {steps.map((val, idx) => {
          const stepName = `t-${19 - idx}`;
          const isLatest = idx === 19;
          const heightPct = Math.round((val / (Math.max(...steps) || 1)) * 100);

          return (
            <div
              key={idx}
              className="flex-1 flex flex-col items-center gap-1 group relative h-full justify-end"
            >
              {/* Tooltip on hover */}
              <div className="absolute -top-7 hidden group-hover:flex bg-[#0a0d14] border border-cyan-500/50 px-1.5 py-0.5 rounded text-[9px] font-mono text-cyan-300 z-10 whitespace-nowrap shadow-lg">
                {stepName}: {Math.round(val * 1000) / 10}%
              </div>

              {/* Bar */}
              <div
                className={`w-full rounded-t transition-all duration-200 ${
                  isLatest
                    ? 'bg-gradient-to-t from-cyan-500 to-cyan-300 shadow-[0_0_8px_#00f0ff]'
                    : 'bg-[#1b253b] hover:bg-cyan-500/60'
                }`}
                style={{ height: `${Math.max(8, heightPct)}%` }}
              />

              {/* Step label (show every 4th label to save space) */}
              <span
                className={`text-[8px] font-mono ${
                  isLatest ? 'text-cyan-300 font-bold' : idx % 4 === 0 ? 'text-slate-400' : 'text-transparent'
                }`}
              >
                {stepName}
              </span>
            </div>
          );
        })}
      </div>
      <div className="flex items-center justify-between text-[10px] font-mono text-slate-500 mt-2 pt-2 border-t border-[#1f293d]">
        <span>t-19 (Oldest Historical Flow in Window)</span>
        <span>t-0 (Current Flow at Observation T)</span>
      </div>
    </div>
  );
};
